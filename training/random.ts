export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    if (!Number.isSafeInteger(seed)) {
      throw new RangeError("seed must be a safe integer");
    }
    this.state = seed >>> 0;
  }

  nextUint32(): number {
    // Mulberry32: compact, deterministic, and sufficient for reproducible
    // simulation. It is intentionally not used for production online deals.
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  }

  next(): number {
    return this.nextUint32() / 0x1_0000_0000;
  }

  int(maxExclusive: number): number {
    if (!Number.isInteger(maxExclusive) || maxExclusive <= 0) {
      throw new RangeError("maxExclusive must be a positive integer");
    }
    return Math.floor(this.next() * maxExclusive);
  }

  shuffle<T>(items: readonly T[]): T[] {
    const result = [...items];
    for (let index = result.length - 1; index > 0; index -= 1) {
      const target = this.int(index + 1);
      [result[index], result[target]] = [result[target], result[index]];
    }
    return result;
  }
}

