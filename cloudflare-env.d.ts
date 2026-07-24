interface Fetcher {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

interface D1ResultMeta {
  changes?: number;
  [key: string]: unknown;
}

interface D1Result<T = unknown> {
  results?: T[];
  success: boolean;
  error?: string;
  meta: D1ResultMeta;
}

interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement;
  first<T = Record<string, unknown>>(columnName?: string): Promise<T | null>;
  run<T = Record<string, unknown>>(): Promise<D1Result<T>>;
  all<T = Record<string, unknown>>(): Promise<D1Result<T>>;
  raw<T = unknown[]>(options?: { columnNames?: boolean }): Promise<T[]>;
}

interface D1Database {
  prepare(query: string): D1PreparedStatement;
  batch<T = unknown>(
    statements: D1PreparedStatement[],
  ): Promise<Array<D1Result<T>>>;
  exec(query: string): Promise<{
    count: number;
    duration: number;
  }>;
}

declare module "cloudflare:workers" {
  export const env: {
    DB?: D1Database;
    [binding: string]: unknown;
  };
}
