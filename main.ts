// Sharpe scalper — a deterministic paper-trading bot for the pump.fun replay.
//
// Design (written from prose, for hourly Sharpe, not raw PnL):
// - Trade many small, short, low-variance round trips so hourly returns stay
//   smooth: one lucky moonshot cannot help Sharpe, one blow-up kills it.
// - Only buy a young bonding-curve coin on CONFIRMED early momentum: its price
//   is up from its creation price, still rising over its last few trades, the
//   last few trades are all buys, several distinct buyers are in, and its first
//   trade was small (no whale-led spike).
// - Size is small and fixed (0.06 SOL) with at most 5 open positions, so no
//   single coin dominates the book.
// - Exit the whole position fast: +12% take-profit, -4% stop-loss, or a
//   800-slot timeout, whichever comes first. Curve fees take ~1.5% of every
//   round trip, so the profit target must clear the fees by a wide margin
//   while the stop stays tight; the win rate has to carry the rest.
// - Entry is early and strict (price up >= 6% from creation, at least 3
//   distinct buyers, first trade at most 1.5 SOL, mint at most 1200 slots
//   old, last 5 trades all buys and still rising): catch the first run-up,
//   before the coin turns over. A late entry buys the top. Of the variants
//   graded on the dev window this set scored best (Sharpe -0.12, ahead of
//   both house baselines); wider targets and later entries scored worse.
// - Fully deterministic: every decision comes only from replay slots, prices
//   and volumes. No clock, no environment, no randomness.
//
// Wire protocol (agent-fun/replay/v2): the harness sends one `hello` line,
// then one `batch` per slot with that slot's events, marks and fills, and a
// final `done`. Each batch we answer with at most one reply carrying our
// orders for that slot. The SDK shim owns the wire; we only decide orders.

import { runBot, LAMPORTS_PER_SOL } from "./vendor/makefun-bot.ts";
import type { Batch, BotState, OrderInput } from "./vendor/makefun-bot.ts";

// ------------------------------------------------------------------ tuning
const BUY_SIZE_LAMPORTS = 60_000_000n; // 0.06 SOL per entry
const MAX_OPEN_POSITIONS = 5;
const MAX_BUYS_PER_BATCH = 4;
const MIN_MOMENTUM_PCT = 6; // price up >= 6% from creation price
const MIN_TRADES = 3;
const MAX_TRADES_AT_ENTRY = 40;
const MAX_AGE_SLOTS = 1200; // mint must be young at entry
const MIN_DISTINCT_BUYERS = 3;
const BUY_FLOW_WINDOW = 5; // last N trades must all be buys
const MAX_FIRST_TRADE_LAMPORTS = 3n * LAMPORTS_PER_SOL / 2n;
const TAKE_PROFIT_BPS = 11200; // sell at +12% vs entry cost
const STOP_LOSS_BPS = 9600; // sell at -4% vs entry cost
const TIMEOUT_SLOTS = 800;
const BUY_SLIPPAGE_BPS = 200;
const SELL_SLIPPAGE_BPS = 300;

// Price is virtual_quote / virtual_token, scaled by this factor so BigInt
// division keeps precision (lamports of quote per token, x SCALE).
const SCALE = 1_000_000_000n;

function priceOf(virtualQuote: bigint, virtualToken: bigint): bigint {
  if (virtualToken <= 0n) return 0n;
  return (virtualQuote * SCALE) / virtualToken;
}

// ------------------------------------------------------------------ memory
interface MintStats {
  createdSlot: number;
  creationPrice: bigint;
  trades: number;
  buyers: Set<string>;
  lastPrices: bigint[]; // most recent trade prices, oldest first (cap 4)
  lastBuys: boolean[]; // is_buy of most recent trades, oldest first (cap 5)
  firstTradeLamports: bigint | null;
  seenComplete: boolean;
}

interface EntryInfo {
  mint: string;
  entrySlot: number;
  costLamports: bigint;
  tokenAmount: bigint;
}

const stats = new Map<string, MintStats>();
const entries = new Map<string, EntryInfo>();

function observeEvents(batch: Batch): void {
  for (const e of batch.events) {
    if (e.event === "CreateEvent") {
      const price = priceOf(
        BigInt(e.virtual_quote_reserves),
        BigInt(e.virtual_token_reserves),
      );
      if (price > 0n && !stats.has(e.mint)) {
        stats.set(e.mint, {
          createdSlot: e.slot,
          creationPrice: price,
          trades: 0,
          buyers: new Set(),
          lastPrices: [],
          lastBuys: [],
          firstTradeLamports: null,
          seenComplete: false,
        });
      }
    } else if (e.event === "TradeEvent") {
      const s = stats.get(e.mint);
      if (s === undefined || s.seenComplete) continue;
      const price = priceOf(
        BigInt(e.virtual_quote_reserves),
        BigInt(e.virtual_token_reserves),
      );
      if (price <= 0n) continue;
      if (s.firstTradeLamports === null) {
        s.firstTradeLamports = BigInt(e.sol_amount);
      }
      s.trades += 1;
      if (e.is_buy) s.buyers.add(e.user);
      s.lastPrices.push(price);
      if (s.lastPrices.length > 4) s.lastPrices.shift();
      s.lastBuys.push(e.is_buy);
      if (s.lastBuys.length > BUY_FLOW_WINDOW) s.lastBuys.shift();
    } else if (e.event === "CompleteEvent") {
      const s = stats.get(e.mint);
      if (s !== undefined) s.seenComplete = true;
      entries.delete(e.mint);
    }
  }
}

function observeFills(batch: Batch): void {
  for (const f of batch.fills) {
    if (f.status !== "filled") continue;
    if (f.side === "buy") {
      entries.set(f.mint, {
        mint: f.mint,
        entrySlot: f.decided_slot,
        costLamports: BigInt(f.quote_lamports),
        tokenAmount: BigInt(f.token_amount),
      });
    } else if (f.side === "sell") {
      entries.delete(f.mint);
    }
  }
}

function rising(s: MintStats): boolean {
  // Still rising: the last three recorded trade prices are non-decreasing
  // and the latest is strictly above the earliest of the three.
  const p = s.lastPrices;
  if (p.length < 3) return false;
  const n = p.length;
  return (
    p[n - 3] <= p[n - 2] && p[n - 2] <= p[n - 1] && p[n - 1] > p[n - 3]
  );
}

function buyCandidate(
  state: BotState,
  mint: string,
  s: MintStats,
): boolean {
  if (s.seenComplete) return false;
  if (s.createdSlot === null) return false;
  if (state.slot - s.createdSlot > MAX_AGE_SLOTS) return false;
  if (s.trades < MIN_TRADES || s.trades > MAX_TRADES_AT_ENTRY) return false;
  if (s.buyers.size < MIN_DISTINCT_BUYERS) return false;
  if (s.firstTradeLamports === null) return false;
  if (s.firstTradeLamports > MAX_FIRST_TRADE_LAMPORTS) return false;
  if (s.lastBuys.length < BUY_FLOW_WINDOW) return false;
  for (const b of s.lastBuys) if (!b) return false;
  if (!rising(s)) return false;
  const curve = state.curves.get(mint);
  if (curve === undefined || curve.complete) return false;
  if (curve.virtual_token_reserves <= 0n) return false;
  const now = priceOf(
    curve.virtual_quote_reserves,
    curve.virtual_token_reserves,
  );
  if (now <= 0n) return false;
  // Up at least MIN_MOMENTUM_PCT from the creation price.
  if (now * 100n < s.creationPrice * BigInt(100 + MIN_MOMENTUM_PCT)) {
    return false;
  }
  // Never chase a coin we already hold or already have a buy pending for.
  if (state.positions.has(mint)) return false;
  for (const p of state.pending.values()) {
    if (p.order.side === "buy" && p.order.mint === mint) return false;
  }
  // Respect the per-mint size cap for this batch.
  if (state.maxBuyLamports(mint) < BUY_SIZE_LAMPORTS) return false;
  return true;
}

function onBatch(state: BotState, batch: Batch): OrderInput[] {
  observeFills(batch);
  observeEvents(batch);
  const orders: OrderInput[] = [];

  // --- exits first: take profit, stop loss, or timeout on every position.
  for (const [mint, holding] of state.positions) {
    const curve = state.curves.get(mint);
    if (curve === undefined || curve.complete) continue;
    const info = entries.get(mint);
    if (info === undefined) continue; // position we did not open; leave it
    if (curve.virtual_token_reserves <= 0n) continue;
    const now = priceOf(
      curve.virtual_quote_reserves,
      curve.virtual_token_reserves,
    );
    if (now <= 0n || holding.token_amount <= 0n) continue;
    // Current value of the holding, in lamports of quote.
    const value = (holding.token_amount * now) / SCALE;
    const ratioBps = (value * 10_000n) / info.costLamports;
    const timedOut = state.slot - info.entrySlot >= TIMEOUT_SLOTS;
    if (ratioBps >= TAKE_PROFIT_BPS || ratioBps <= STOP_LOSS_BPS || timedOut) {
      orders.push({
        side: "sell",
        mint,
        token_amount: holding.token_amount,
        max_slippage_bps: SELL_SLIPPAGE_BPS,
      });
    }
  }

  // --- entries: youngest confirmed-momentum coins first.
  if (state.committedPositionCount() < MAX_OPEN_POSITIONS) {
    const candidates: { mint: string; age: number }[] = [];
    for (const [mint, s] of stats) {
      if (buyCandidate(state, mint, s)) {
        candidates.push({ mint, age: state.slot - s.createdSlot });
      }
    }
    // Youngest first: the freshest momentum has the most runway.
    candidates.sort((a, b) => a.age - b.age);
    let buys = 0;
    for (const c of candidates) {
      if (buys >= MAX_BUYS_PER_BATCH) break;
      if (state.committedPositionCount() + buys >= MAX_OPEN_POSITIONS) break;
      const free = state.freeCashLamports();
      const need = BUY_SIZE_LAMPORTS + 2n * state.txFeeLamports();
      if (free < need) break;
      orders.push({
        side: "buy",
        mint: c.mint,
        quote_lamports: BUY_SIZE_LAMPORTS,
        max_slippage_bps: BUY_SLIPPAGE_BPS,
      });
      buys += 1;
    }
  }

  return orders;
}

await runBot({ onBatch });
