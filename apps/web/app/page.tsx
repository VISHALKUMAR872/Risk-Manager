"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useRef,
  type ReactNode,
  type MouseEvent as ReactMouseEvent,
} from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Clock3,
  RefreshCw,
  ShieldCheck,
  ShieldAlert,
  Search,
  XCircle,
  Network,
  UserRound,
  Monitor,
  Globe2,
  CreditCard,
  Store,
  CircleDot,
  Link2,
  Maximize2,
  X,
} from "lucide-react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type DashboardTransaction = {
  transaction_id: string;
  event_id: string;
  event_time: string;
  customer_id: string;
  merchant_id: string;
  amount: string;
  currency: string;
  device_id: string;
  ip_address: string;
  payment_method: string;
  merchant_category: string;
  country: string;
  channel: string;
  status: string;
  created_at: string;

  fraud_probability: number | null;
  expected_loss: number | null;
  risk_level: string | null;
  decision: string | null;
  reason_codes: string[];
  policy_version: string | null;
  model_version: string | null;
  calibration_version: string | null;
};
type DashboardSummary = {
  transaction_count: number;
  risk_decided_count: number;
  pending_count: number;
  failed_count: number;

  intervention_count: number;
  expected_loss: number;
  high_risk_count: number;

  decisions: {
    APPROVE: number;
    VERIFY: number;
    REVIEW: number;
    HOLD: number;
  };

  risk_levels: {
    LOW: number;
    MEDIUM: number;
    HIGH: number;
    CRITICAL: number;
  };

  average_transaction_amount: number;
};
type NetworkNode = {
  id: string;
  type: string;
  label: string;
  selected: boolean;
  amount: number | null;
  currency: string | null;
  event_time: string | null;
};

type NetworkEdge = {
  source: string;
  target: string;
  type: string;
};

type TransactionNetwork = {
  transaction_id: string;
  nodes: NetworkNode[];
  edges: NetworkEdge[];
};

function formatCurrency(amount: number, currency = "INR") {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(amount);
}

function formatTime(value: string) {
  return new Date(value).toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function shortId(value: string, maxLength = 18) {
  if (value.length <= maxLength) return value;

  const suffixLength = Math.min(6, Math.max(3, Math.floor(maxLength / 3)));
  const prefixLength = Math.max(
    1,
    maxLength - suffixLength - 1,
  );

  return `${value.slice(0, prefixLength)}…${value.slice(-suffixLength)}`;
}

function riskClass(level?: string | null) {
  switch (level) {
    case "CRITICAL":
      return "risk-critical";
    case "HIGH":
      return "risk-high";
    case "MEDIUM":
      return "risk-medium";
    default:
      return "risk-low";
  }
}

function decisionIcon(decision?: string | null) {
  switch (decision) {
    case "HOLD":
      return <XCircle size={15} />;
    case "REVIEW":
      return <AlertTriangle size={15} />;
    case "VERIFY":
      return <ShieldAlert size={15} />;
    default:
      return <CheckCircle2 size={15} />;
  }
}

function networkNodeIcon(type: string) {
  switch (type) {
    case "CUSTOMER":
      return <UserRound size={13} />;
    case "TRANSACTION":
      return <CircleDot size={13} />;
    case "RELATED_TRANSACTION":
      return <CircleDot size={13} />;
    case "MERCHANT":
      return <Store size={13} />;
    case "DEVICE":
      return <Monitor size={13} />;
    case "IP":
      return <Globe2 size={13} />;
    case "PAYMENT":
      return <CreditCard size={13} />;
    default:
      return <CircleDot size={13} />;
  }
}

export default function Home() {
  const [transactions, setTransactions] = useState<
    DashboardTransaction[]
  >([]);

  const [selected, setSelected] =
    useState<DashboardTransaction | null>(null);
  const [summary, setSummary] =
    useState<DashboardSummary | null>(null);
  const [network, setNetwork] =
    useState<TransactionNetwork | null>(null);

  const [networkLoading, setNetworkLoading] =
    useState(false);

  const [networkError, setNetworkError] =
    useState("");

  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] =
    useState<Date | null>(null);

  const [networkExpanded, setNetworkExpanded] =
    useState(false);

  const loadTransactions = useCallback(async () => {
    try {
      setError("");

      const response = await fetch(
        `${API_URL}/transactions/dashboard?limit=50`,
        {
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Dashboard API returned ${response.status}`
        );
      }

      const data: DashboardTransaction[] =
        await response.json();

      const summaryResponse = await fetch(
        `${API_URL}/transactions/dashboard/summary`,
        {
          cache: "no-store",
        }
      );

      if (!summaryResponse.ok) {
        throw new Error(
          `Dashboard summary API returned ${summaryResponse.status}`
        );
      }

      const summaryData: DashboardSummary =
        await summaryResponse.json();

      setSummary(summaryData);
      setTransactions(data);
      setLastUpdated(new Date());

      setSelected((current) => {
        if (!current) {
          return data[0] ?? null;
        }

        const refreshedSelected =
          data.find(
            (transaction) =>
              transaction.transaction_id ===
              current.transaction_id,
          );

        /*
         * Dashboard polling happens every 5 seconds.
         * Keep the same selected object when its transaction
         * is still present. Otherwise React treats the selection
         * as changed, which triggers the Neo4j network effect and
         * causes the graph/modal to visibly blink.
         */
        if (refreshedSelected) {
          return current;
        }

        return data[0] ?? null;
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load dashboard data"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void loadTransactions();
    }, 0);

    const interval = window.setInterval(
      loadTransactions,
      5000
    );

    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(interval);
    };
  }, [loadTransactions]);

  const selectNetworkTransaction = useCallback(
    async (transactionId: string) => {
      const cached = transactions.find(
        (item) =>
          item.transaction_id === transactionId,
      );

      if (cached) {
        setSelected(cached);
        return;
      }

      try {
        const [transactionResponse, riskResponse] =
          await Promise.all([
            fetch(
              `${API_URL}/transactions/${encodeURIComponent(
                transactionId,
              )}`,
              { cache: "no-store" },
            ),
            fetch(
              `${API_URL}/transactions/${encodeURIComponent(
                transactionId,
              )}/risk`,
              { cache: "no-store" },
            ),
          ]);

        if (
          !transactionResponse.ok ||
          !riskResponse.ok
        ) {
          throw new Error(
            "Unable to load the related transaction.",
          );
        }

        const transaction =
          await transactionResponse.json();

        const risk =
          await riskResponse.json();

        setSelected({
          ...transaction,
          fraud_probability:
            risk.fraud_probability ?? null,
          expected_loss:
            risk.expected_loss ?? null,
          risk_level:
            risk.risk_level ?? null,
          decision:
            risk.decision ?? null,
          reason_codes:
            risk.reason_codes ?? [],
          policy_version:
            risk.policy_version ?? null,
          model_version:
            risk.model_version ?? null,
          calibration_version:
            risk.calibration_version ?? null,
        });
      } catch (err) {
        setNetworkError(
          err instanceof Error
            ? err.message
            : "Unable to load the related transaction.",
        );
      }
    },
    [transactions],
  );

  const networkRequestId = useRef(0);

  const loadNetwork = useCallback(
    async (transactionId: string) => {
      const requestId =
        ++networkRequestId.current;

      try {
        setNetworkLoading(true);
        setNetworkError("");

        const response = await fetch(
          `${API_URL}/transactions/${encodeURIComponent(
            transactionId
          )}/network`,
          {
            cache: "no-store",
          }
        );

        if (response.status === 404) {
          setNetworkError(
            "No entity network is available for this transaction."
          );
          return;
        }

        if (!response.ok) {
          throw new Error(
            `Network API returned ${response.status}`
          );
        }

        const data: TransactionNetwork =
          await response.json();

        if (
          requestId !==
          networkRequestId.current
        ) {
          return;
        }

        setNetwork(data);
      } catch (err) {
        if (
          requestId !==
          networkRequestId.current
        ) {
          return;
        }

        setNetworkError(
          err instanceof Error
            ? err.message
            : "Unable to load entity network"
        );
      } finally {
        if (
          requestId ===
          networkRequestId.current
        ) {
          setNetworkLoading(false);
        }
      }
    },
    []
  );

  useEffect(() => {
    if (!selected) {
      networkRequestId.current += 1;
      return;
    }

    const networkLoad = window.setTimeout(() => {
      void loadNetwork(selected.transaction_id);
    }, 0);

    return () => {
      window.clearTimeout(networkLoad);
    };
  }, [selected, loadNetwork]);

  useEffect(() => {
    if (!networkExpanded) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setNetworkExpanded(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () =>
      window.removeEventListener(
        "keydown",
        handleKeyDown,
      );
  }, [networkExpanded]);

  const filteredTransactions = useMemo(() => {
    const query = search.toLowerCase().trim();

    if (!query) return transactions;

    return transactions.filter((transaction) =>
      [
        transaction.transaction_id,
        transaction.customer_id,
        transaction.merchant_id,
        transaction.device_id,
        transaction.ip_address,
        transaction.payment_method,
        transaction.decision,
        transaction.risk_level,
        transaction.status,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  }, [transactions, search]);

  return (
    <main className="min-h-screen bg-[#080b10] text-white">
      <header className="border-b border-white/10 bg-[#0b0f15]">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-400/10 ring-1 ring-emerald-400/20">
              <ShieldCheck
                className="text-emerald-400"
                size={22}
              />
            </div>

            <div>
              <div className="text-lg font-semibold tracking-tight">
                Risk Sentinel
              </div>

              <div className="text-xs text-slate-500">
                Real-time fraud loss prevention
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2 text-slate-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.7)]" />
              Live
            </div>

            {lastUpdated && (
              <div className="hidden text-slate-500 sm:block">
                Updated{" "}
                {formatTime(lastUpdated.toISOString())}
              </div>
            )}

            <button
              onClick={loadTransactions}
              className="rounded-lg border border-white/10 p-2 text-slate-400 transition hover:bg-white/5 hover:text-white"
              title="Refresh"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] px-6 py-6">
        <section className="mb-6">
          <div className="mb-5">
            <p className="mb-1 text-xs font-medium uppercase tracking-[0.2em] text-emerald-400">
              Risk Operations
            </p>

            <h1 className="text-2xl font-semibold tracking-tight">
              Command Center
            </h1>

            <p className="mt-1 text-sm text-slate-500">
              Monitor transactions, intervention decisions, and expected loss.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Transactions"
              value={
                summary
                  ? summary.transaction_count.toLocaleString("en-IN")
                  : "—"
              }
              icon={<ArrowUpRight size={18} />}
            />

            <MetricCard
              label="Interventions"
              value={
                summary
                  ? summary.intervention_count.toLocaleString("en-IN")
                  : "—"
              }
              icon={<ShieldAlert size={18} />}
              accent
            />

            <MetricCard
              label="Expected Loss"
              value={
                summary
                  ? formatCurrency(summary.expected_loss)
                  : "—"
              }
              icon={<Clock3 size={18} />}
            />

            <MetricCard
              label="High Risk"
              value={
                summary
                  ? summary.high_risk_count.toLocaleString("en-IN")
                  : "—"
              }
              icon={<AlertTriangle size={18} />}
              danger={
                summary
                  ? summary.high_risk_count > 0
                  : false
              }
            />
          </div>
        </section>

        {summary && (
          <div className="mb-5 grid grid-cols-3 gap-3">
            <StatusMetric
              label="Risk Decided"
              value={summary.risk_decided_count}
            />

            <StatusMetric
              label="Pending"
              value={summary.pending_count}
              warning={summary.pending_count > 0}
            />

            <StatusMetric
              label="Failed"
              value={summary.failed_count}
              danger={summary.failed_count > 0}
            />
          </div>
        )}

        {summary && (
          <section className="mb-5 grid gap-5 lg:grid-cols-2">
            <DistributionCard
              title="Decision Mix"
              total={summary.risk_decided_count}
              items={[
                {
                  label: "APPROVE",
                  value: summary.decisions.APPROVE,
                  className: "bg-emerald-400",
                },
                {
                  label: "VERIFY",
                  value: summary.decisions.VERIFY,
                  className: "bg-amber-400",
                },
                {
                  label: "REVIEW",
                  value: summary.decisions.REVIEW,
                  className: "bg-orange-400",
                },
                {
                  label: "HOLD",
                  value: summary.decisions.HOLD,
                  className: "bg-red-400",
                },
              ]}
            />

            <DistributionCard
              title="Risk Distribution"
              total={summary.risk_decided_count}
              items={[
                {
                  label: "LOW",
                  value: summary.risk_levels.LOW,
                  className: "bg-emerald-400",
                },
                {
                  label: "MEDIUM",
                  value: summary.risk_levels.MEDIUM,
                  className: "bg-amber-400",
                },
                {
                  label: "HIGH",
                  value: summary.risk_levels.HIGH,
                  className: "bg-orange-400",
                },
                {
                  label: "CRITICAL",
                  value: summary.risk_levels.CRITICAL,
                  className: "bg-red-400",
                },
              ]}
            />
          </section>
        )}

        {error && (
          <div className="mb-5 flex items-center gap-3 rounded-xl border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-300">
            <AlertTriangle size={17} />
            <span>{error}</span>
          </div>
        )}

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_450px]">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#0d1219]">
            <div className="flex flex-col gap-3 border-b border-white/10 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="font-medium">
                  Transaction Stream
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  Latest persisted decisions
                </p>
              </div>

              <div className="relative">
                <Search
                  size={15}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                />

                <input
                  value={search}
                  onChange={(event) =>
                    setSearch(event.target.value)
                  }
                  placeholder="Search transactions..."
                  className="w-full rounded-lg border border-white/10 bg-[#080b10] py-2 pl-9 pr-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-emerald-400/40 sm:w-64"
                />
              </div>
            </div>

            {loading ? (
              <div className="flex h-72 items-center justify-center text-sm text-slate-500">
                Loading transaction stream...
              </div>
            ) : filteredTransactions.length === 0 ? (
              <div className="flex h-72 items-center justify-center text-sm text-slate-500">
                No transactions found.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[850px] text-left text-sm">
                  <thead className="border-b border-white/10 bg-white/[0.02] text-xs uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="px-4 py-3 font-medium">
                        Transaction
                      </th>

                      <th className="px-4 py-3 font-medium">
                        Amount
                      </th>

                      <th className="px-4 py-3 font-medium">
                        Time
                      </th>

                      <th className="px-4 py-3 font-medium">
                        Risk
                      </th>

                      <th className="px-4 py-3 font-medium">
                        Decision
                      </th>

                      <th className="px-4 py-3 text-right font-medium">
                        Expected Loss
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-white/[0.06]">
                    {filteredTransactions.map(
                      (transaction) => (
                        <tr
                          key={transaction.transaction_id}
                          onClick={() =>
                            setSelected(transaction)
                          }
                          className={`cursor-pointer transition hover:bg-white/[0.035] ${
                            selected?.transaction_id ===
                            transaction.transaction_id
                              ? "bg-emerald-400/[0.04]"
                              : ""
                          }`}
                        >
                          <td className="px-4 py-4">
                            <div className="font-mono text-xs text-slate-300">
                              {shortId(
                                transaction.transaction_id
                              )}
                            </div>

                            <div className="mt-1 text-xs text-slate-600">
                              {transaction.merchant_category}
                            </div>
                          </td>

                          <td className="px-4 py-4 font-medium">
                            {formatCurrency(
                              Number(transaction.amount),
                              transaction.currency
                            )}
                          </td>

                          <td className="px-4 py-4 text-xs text-slate-400">
                            {formatTime(
                              transaction.event_time
                            )}
                          </td>

                          <td className="px-4 py-4">
                            <span
                              className={`risk-badge ${riskClass(
                                transaction.risk_level
                              )}`}
                            >
                              {transaction.risk_level ??
                                "PENDING"}
                            </span>
                          </td>

                          <td className="px-4 py-4">
                            <span className="flex items-center gap-1.5 text-xs font-medium text-slate-300">
                              {decisionIcon(
                                transaction.decision
                              )}

                              {transaction.decision ??
                                transaction.status}
                            </span>
                          </td>

                          <td className="px-4 py-4 text-right font-mono text-xs text-slate-300">
                            {transaction.expected_loss !==
                            null
                              ? formatCurrency(
                                  transaction.expected_loss
                                )
                              : "—"}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <TransactionDetail
            transaction={selected}
            network={network}
            networkLoading={networkLoading}
            networkError={networkError}
            onExpandNetwork={() =>
              setNetworkExpanded(true)
            }
            onSelectTransaction={
              selectNetworkTransaction
            }
          />
        </section>
      </div>

      {networkExpanded &&
        selected &&
        network &&
        !networkLoading && (
          <NetworkModal
            transaction={selected}
            network={network}
            onSelectTransaction={
              selectNetworkTransaction
            }
            onClose={() =>
              setNetworkExpanded(false)
            }
          />
        )}
    </main>
  );
}

function StatusMetric({
  label,
  value,
  warning = false,
  danger = false,
}: {
  label: string;
  value: number;
  warning?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0d1219] px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-slate-600">
          {label}
        </span>

        <span
          className={`text-sm font-semibold ${
            danger
              ? "text-red-400"
              : warning
                ? "text-amber-400"
                : "text-emerald-400"
          }`}
        >
          {value.toLocaleString("en-IN")}
        </span>
      </div>
    </div>
  );
}

function DistributionCard({
  title,
  total,
  items,
}: {
  title: string;
  total: number;
  items: {
    label: string;
    value: number;
    className: string;
  }[];
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#0d1219] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-slate-500">
            {title}
          </p>

          <p className="mt-1 text-[10px] text-slate-600">
            {total.toLocaleString("en-IN")} risk-decided transactions
          </p>
        </div>
      </div>

      <div className="space-y-3">
        {items.map((item) => {
          const percentage =
            total > 0 ? (item.value / total) * 100 : 0;

          return (
            <div key={item.label}>
              <div className="mb-1.5 flex items-center justify-between text-[11px]">
                <span className="font-medium text-slate-400">
                  {item.label}
                </span>

                <span className="font-mono text-slate-500">
                  {item.value.toLocaleString("en-IN")}
                  {" · "}
                  {percentage.toFixed(1)}%
                </span>
              </div>

              <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
                <div
                  className={`h-full rounded-full ${item.className}`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon,
  accent = false,
  danger = false,
}: {
  label: string;
  value: string;
  icon: ReactNode;
  accent?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#0d1219] p-5">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-slate-500">
          {label}
        </span>

        <div
          className={`rounded-lg p-2 ${
            danger
              ? "bg-red-400/10 text-red-400"
              : accent
                ? "bg-amber-400/10 text-amber-400"
                : "bg-white/5 text-slate-400"
          }`}
        >
          {icon}
        </div>
      </div>

      <div className="text-2xl font-semibold tracking-tight">
        {value}
      </div>
    </div>
  );
}

function TransactionDetail({
  transaction,
  network,
  networkLoading,
  networkError,
  onExpandNetwork,
  onSelectTransaction,
}: {
  transaction: DashboardTransaction | null;
  network: TransactionNetwork | null;
  networkLoading: boolean;
  networkError: string;
  onExpandNetwork: () => void;
  onSelectTransaction?: (
    transactionId: string,
  ) => void | Promise<void>;
}) {
  if (!transaction) {
    return (
      <aside className="rounded-2xl border border-white/10 bg-[#0d1219] p-6">
        <div className="flex h-full min-h-[500px] items-center justify-center text-center">
          <div>
            <ShieldCheck
              size={32}
              className="mx-auto mb-3 text-slate-700"
            />

            <p className="text-sm text-slate-500">
              Select a transaction to investigate.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="rounded-2xl border border-white/10 bg-[#0d1219]">
      <div className="border-b border-white/10 p-5">
        <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
          Investigation
        </p>

        <div className="mt-2 flex items-start justify-between gap-3">
          <div>
            <h2 className="font-semibold">
              Transaction Detail
            </h2>

            <p className="mt-1 break-all font-mono text-[11px] text-slate-600">
              {transaction.transaction_id}
            </p>
          </div>

          <span
            className={`risk-badge ${riskClass(
              transaction.risk_level
            )}`}
          >
            {transaction.risk_level ?? "PENDING"}
          </span>
        </div>
      </div>

      <div className="space-y-5 p-5">
        <div className="rounded-xl border border-white/10 bg-[#080b10] p-4">
          <p className="text-xs text-slate-500">
            Transaction amount
          </p>

          <p className="mt-1 text-2xl font-semibold">
            {formatCurrency(
              Number(transaction.amount),
              transaction.currency
            )}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <DetailMetric
            label="Fraud Probability"
            value={
              transaction.fraud_probability !== null
                ? `${(
                    transaction.fraud_probability * 100
                  ).toFixed(2)}%`
                : "—"
            }
          />

          <DetailMetric
            label="Expected Loss"
            value={
              transaction.expected_loss !== null
                ? formatCurrency(
                    transaction.expected_loss
                  )
                : "—"
            }
          />
        </div>

        <div>
          <p className="mb-2 text-xs uppercase tracking-wider text-slate-500">
            Decision
          </p>

          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-4 py-3">
            {decisionIcon(transaction.decision)}

            <span className="font-medium">
              {transaction.decision ?? "Unavailable"}
            </span>
          </div>
        </div>

        <div>
          <p className="mb-2 text-xs uppercase tracking-wider text-slate-500">
            Reason codes
          </p>

          <div className="space-y-2">
            {transaction.reason_codes.map(
              (reason) => (
                <div
                  key={reason}
                  className="rounded-lg border border-amber-400/10 bg-amber-400/5 px-3 py-2 text-xs text-amber-300"
                >
                  {reason.replaceAll("_", " ")}
                </div>
              )
            )}

            {!transaction.reason_codes.length && (
              <p className="text-xs text-slate-600">
                No reason codes available.
              </p>
            )}
          </div>
        </div>

        <div>
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs uppercase tracking-wider text-slate-500">
              Entity context
            </p>

            <Network
              size={14}
              className="text-slate-600"
            />
          </div>

          <div className="space-y-2 text-xs">
            <InfoRow
              label="Customer"
              value={transaction.customer_id}
            />

            <InfoRow
              label="Merchant"
              value={transaction.merchant_id}
            />

            <InfoRow
              label="Device"
              value={transaction.device_id}
            />

            <InfoRow
              label="IP"
              value={transaction.ip_address}
            />

            <InfoRow
              label="Payment"
              value={transaction.payment_method}
            />

            <InfoRow
              label="Channel"
              value={transaction.channel}
            />

            <InfoRow
              label="Country"
              value={transaction.country}
            />
          </div>
        </div>

        <RiskNetwork
          network={network}
          loading={networkLoading}
          error={networkError}
          onExpand={onExpandNetwork}
          onSelectTransaction={
            onSelectTransaction
          }
        />

        <div className="border-t border-white/10 pt-4">
          <p className="mb-3 text-xs uppercase tracking-wider text-slate-500">
            Decision provenance
          </p>

          <div className="space-y-2 text-xs">
            <InfoRow
              label="Model"
              value={
                transaction.model_version ?? "—"
              }
            />

            <InfoRow
              label="Calibration"
              value={
                transaction.calibration_version ?? "—"
              }
            />

            <InfoRow
              label="Policy"
              value={
                transaction.policy_version ?? "—"
              }
            />

            <InfoRow
              label="Status"
              value={transaction.status}
            />
          </div>
        </div>
      </div>
    </aside>
  );
}

function RiskNetwork({
  network,
  loading,
  error,
  onExpand,
  onSelectTransaction,
}: {
  network: TransactionNetwork | null;
  loading: boolean;
  error: string;
  onExpand: () => void;
  onSelectTransaction?: (
    transactionId: string,
  ) => void | Promise<void>;
}) {
  const relatedTransactions =
    network?.nodes.filter(
      (node) => node.type === "RELATED_TRANSACTION"
    ) ?? [];

  const sharedRelationships = useMemo(() => {
    const relationships = new Set(
      (network?.edges ?? [])
        .map((edge) => edge.type)
        .filter((type) =>
          type.startsWith("SHARED_")
        )
    );

    return {
      customer: relationships.has(
        "SHARED_CUSTOMER"
      ),
      device: relationships.has(
        "SHARED_DEVICE"
      ),
      ip: relationships.has("SHARED_IP"),
      payment: relationships.has(
        "SHARED_PAYMENT"
      ),
      merchant: relationships.has(
        "SHARED_MERCHANT"
      ),
    };
  }, [network]);

  return (
    <div className="border-t border-white/10 pt-5">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wider text-slate-500">
            Risk Network
          </p>

          <p className="mt-1 text-[11px] text-slate-600">
            Live entity relationships from Neo4j
          </p>
        </div>

        <Network
          size={16}
          className="text-emerald-400/70"
        />
      </div>

      {loading ? (
        <div className="flex h-[360px] items-center justify-center rounded-xl border border-white/10 bg-[#080b10]">
          <div className="text-center">
            <RefreshCw
              size={20}
              className="mx-auto mb-2 animate-spin text-slate-600"
            />

            <p className="text-xs text-slate-500">
              Loading entity network...
            </p>
          </div>
        </div>
      ) : error ? (
        <div className="rounded-xl border border-white/10 bg-[#080b10] p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle
              size={15}
              className="mt-0.5 shrink-0 text-slate-600"
            />

            <p className="text-xs leading-5 text-slate-500">
              {error}
            </p>
          </div>
        </div>
      ) : !network || network.nodes.length === 0 ? (
        <div className="rounded-xl border border-white/10 bg-[#080b10] p-4">
          <p className="text-xs text-slate-600">
            No entity network available.
          </p>
        </div>
      ) : (
        <>
          <button
            type="button"
            onClick={onExpand}
            className="group relative block w-full overflow-hidden rounded-xl border border-white/10 bg-[#080b10] text-left transition hover:border-emerald-400/20 focus:outline-none focus:ring-1 focus:ring-emerald-400/40"
            title="Open network in full-screen investigation view"
          >
            <NetworkGraph
              network={network}
              onSelectTransaction={
                onSelectTransaction
              }
            />

            <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-lg border border-white/10 bg-[#0b0f15]/90 px-2 py-1.5 text-[9px] text-slate-500 backdrop-blur-sm transition group-hover:text-emerald-300">
              <Maximize2 size={11} />
              Expand
            </div>
          </button>

          <div className="mt-3">
            <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-600">
              <Link2 size={11} />
              Network exposure
            </div>

            <div className="grid grid-cols-2 gap-2">
              <ExposureMetric
                label="Related transactions"
                value={relatedTransactions.length}
                highlighted={
                  relatedTransactions.length > 0
                }
              />

              <ExposureMetric
                label="Shared IP"
                value={sharedRelationships.ip ? 1 : 0}
                highlighted={sharedRelationships.ip}
              />

              <ExposureMetric
                label="Shared device"
                value={
                  sharedRelationships.device ? 1 : 0
                }
                highlighted={
                  sharedRelationships.device
                }
              />

              <ExposureMetric
                label="Shared payment"
                value={
                  sharedRelationships.payment ? 1 : 0
                }
                highlighted={
                  sharedRelationships.payment
                }
              />

              <ExposureMetric
                label="Shared customer"
                value={
                  sharedRelationships.customer ? 1 : 0
                }
                highlighted={
                  sharedRelationships.customer
                }
              />

              <ExposureMetric
                label="Shared merchant"
                value={
                  sharedRelationships.merchant ? 1 : 0
                }
                highlighted={
                  sharedRelationships.merchant
                }
              />
            </div>
          </div>

          {relatedTransactions.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-[10px] uppercase tracking-wider text-slate-600">
                Connected transactions
              </p>

              <div className="space-y-1.5">
                {relatedTransactions.map(
                  (transaction) => (
                    <div
                      key={transaction.id}
                      role="button"
                      tabIndex={0}
                      onClick={() =>
                        onSelectTransaction?.(
                          transaction.label,
                        )
                      }
                      onKeyDown={(event) => {
                        if (
                          event.key === "Enter" ||
                          event.key === " "
                        ) {
                          event.preventDefault();
                          onSelectTransaction?.(
                            transaction.label,
                          );
                        }
                      }}
                      className="flex cursor-pointer items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.015] px-3 py-2 text-left transition hover:border-amber-400/25 hover:bg-amber-400/[0.03]"
                    >
                      <div className="min-w-0">
                        <p className="truncate font-mono text-[10px] text-slate-400">
                          {shortId(transaction.label)}
                        </p>

                        {transaction.event_time && (
                          <p className="mt-0.5 text-[9px] text-slate-600">
                            {formatTime(
                              transaction.event_time
                            )}
                          </p>
                        )}
                      </div>

                      <span className="ml-3 shrink-0 text-[10px] font-medium text-slate-400">
                        {transaction.amount !== null
                          ? formatCurrency(
                              transaction.amount,
                              transaction.currency ??
                                "INR"
                            )
                          : "—"}
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function NetworkGraph({
  network,
  expanded = false,
  onSelectTransaction,
}: {
  network: TransactionNetwork;
  expanded?: boolean;
  onSelectTransaction?: (
    transactionId: string,
  ) => void | Promise<void>;
}) {
  const relatedNodes = network.nodes.filter(
    (node) => node.type === "RELATED_TRANSACTION",
  );

  const coreNodes = network.nodes.filter(
    (node) => node.type !== "RELATED_TRANSACTION",
  );

  /*
   * Expanded investigation canvas:
   *
   *       CUSTOMER
   *          |
   * DEVICE -- TX -- IP ===== shared-entity lane =====>
   *          |
   *       MERCHANT
   *          |
   *       PAYMENT
   *
   * Related transactions live in their own lower grid.
   * Shared paths are routed through a dedicated right/bottom
   * corridor so they never cross node labels.
   */

  const canvasWidth = expanded ? 900 : 700;

  const positions: Record<
    string,
    { x: number; y: number }
  > = expanded
    ? {
        CUSTOMER: {
          x: 390,
          y: 90,
        },
        DEVICE: {
          x: 150,
          y: 235,
        },
        TRANSACTION: {
          x: 390,
          y: 235,
        },
        IP: {
          x: 650,
          y: 235,
        },
        MERCHANT: {
          x: 390,
          y: 375,
        },
        PAYMENT: {
          x: 390,
          y: 480,
        },
      }
    : {
        CUSTOMER: {
          x: 350,
          y: 80,
        },
        DEVICE: {
          x: 120,
          y: 225,
        },
        TRANSACTION: {
          x: 350,
          y: 225,
        },
        IP: {
          x: 580,
          y: 225,
        },
        MERCHANT: {
          x: 350,
          y: 365,
        },
        PAYMENT: {
          x: 350,
          y: 470,
        },
      };

  const relatedColumns = expanded ? 4 : 3;

  const relatedXPositions = expanded
    ? [120, 340, 560, 780]
    : [105, 350, 595];

  const relatedPositions = relatedNodes.map(
    (node, index) => {
      const column =
        index % relatedColumns;

      const row = Math.floor(
        index / relatedColumns,
      );

      return {
        node,
        x: relatedXPositions[column],
        y:
          (expanded ? 645 : 610) +
          row * (expanded ? 135 : 125),
      };
    },
  );

  const nodePositions = new Map<
    string,
    { x: number; y: number }
  >();

  coreNodes.forEach((node) => {
    const position = positions[node.type];

    if (position) {
      nodePositions.set(
        node.id,
        position,
      );
    }
  });

  relatedPositions.forEach(
    ({ node, x, y }) => {
      nodePositions.set(node.id, {
        x,
        y,
      });
    },
  );

  const directEdges = network.edges.filter(
    (edge) =>
      !edge.type.startsWith("SHARED_"),
  );

  const sharedEdges = network.edges.filter(
    (edge) =>
      edge.type.startsWith("SHARED_"),
  );

  const sharedRelationships = Array.from(
    new Set(
      sharedEdges.map((edge) =>
        edge.type.replace(
          "SHARED_",
          "",
        ),
      ),
    ),
  );

  const relatedRows = Math.max(
    1,
    Math.ceil(
      relatedNodes.length /
        relatedColumns,
    ),
  );

  const graphHeight = expanded
    ? 810 +
      (relatedRows - 1) * 135
    : 750 +
      (relatedRows - 1) * 125;

  function getPosition(nodeId: string) {
    return nodePositions.get(nodeId);
  }

  function getLabel(type: string) {
    switch (type) {
      case "CUSTOMER":
        return "CUSTOMER";
      case "DEVICE":
        return "DEVICE";
      case "IP":
        return "IP ADDRESS";
      case "PAYMENT":
        return "PAYMENT";
      case "MERCHANT":
        return "MERCHANT";
      default:
        return type;
    }
  }

  function getValue(node: NetworkNode) {
    if (
      node.type === "TRANSACTION" ||
      node.type === "RELATED_TRANSACTION"
    ) {
      return shortId(
        node.label,
        expanded ? 20 : 15,
      );
    }

    return shortId(
      node.label,
      expanded ? 22 : 17,
    );
  }

  function directPath(
    source: { x: number; y: number },
    target: { x: number; y: number },
  ) {
    const dx = target.x - source.x;

    if (Math.abs(dx) < 5) {
      return `
        M ${source.x} ${source.y}
        L ${target.x} ${target.y}
      `;
    }

    const midY =
      (source.y + target.y) / 2;

    return `
      M ${source.x} ${source.y}
      C ${source.x} ${midY},
        ${target.x} ${midY},
        ${target.x} ${target.y}
    `;
  }

  /*
   * Shared paths use a dedicated corridor below the core
   * and then approach each related transaction from above.
   * For IP, the source leaves to the right first, keeping
   * the central graph completely unobstructed.
   */
  function sharedPath(
    edgeIndex: number,
    edge: NetworkEdge,
    source: { x: number; y: number },
    target: { x: number; y: number },
  ) {
    const relationship =
      edge.type.replace(
        "SHARED_",
        "",
      );

    if (relationship === "IP") {
      const sideX =
        Math.min(
          canvasWidth - 30,
          source.x +
            45 +
            (edgeIndex % 4) * 16,
        );

      const corridorY =
        545 +
        (edgeIndex % 4) * 12;

      return `
        M ${source.x} ${source.y}
        C ${sideX} ${source.y},
          ${sideX} ${corridorY},
          ${target.x} ${target.y - 38}
      `;
    }

    const corridorY =
      555 +
      (edgeIndex % 4) * 14;

    return `
      M ${source.x} ${source.y}
      C ${source.x} ${corridorY},
        ${target.x} ${corridorY},
        ${target.x} ${target.y - 38}
    `;
  }

  return (
    <div className="overflow-hidden rounded-xl border border-white/10 bg-[#080b10]">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-300">
            Entity Relationship Graph
          </p>

          <p className="mt-1 text-[9px] text-slate-600">
            {expanded
              ? "Expanded investigation view"
              : "Selected transaction and connected entities"}
          </p>
        </div>

        <div className="flex items-center gap-3 text-[8px]">
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            Selected
          </span>

          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
            Related
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${canvasWidth} ${graphHeight}`}
          className={`block h-auto w-full ${
            expanded
              ? "min-w-[860px]"
              : "min-w-[680px]"
          }`}
          preserveAspectRatio="xMidYMin meet"
          role="img"
          aria-label="Transaction entity investigation network"
        >
          <defs>
            <filter
              id="riskTransactionGlow"
              x="-100%"
              y="-100%"
              width="300%"
              height="300%"
            >
              <feGaussianBlur
                stdDeviation="5"
                result="blur"
              />

              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Core region */}
          <rect
            x="18"
            y="18"
            width={canvasWidth - 36}
            height="505"
            rx="18"
            fill="rgba(255,255,255,0.012)"
            stroke="rgba(255,255,255,0.055)"
          />

          <text
            x="38"
            y="45"
            fill="#64748b"
            fontSize="9"
            fontWeight="600"
            letterSpacing="1.3"
          >
            TRANSACTION CONTEXT
          </text>

          {/* Shared-entity corridor label */}
          {sharedEdges.length > 0 && (
            <text
              x={
                expanded
                  ? canvasWidth - 175
                  : canvasWidth - 160
              }
              y="530"
              textAnchor="middle"
              fill="#78551a"
              fontSize="7"
              letterSpacing="0.9"
            >
              SHARED ENTITY PATHS
            </text>
          )}

          {/* Related region */}
          <rect
            x="18"
            y="555"
            width={canvasWidth - 36}
            height={graphHeight - 570}
            rx="18"
            fill="rgba(245,158,11,0.012)"
            stroke="rgba(245,158,11,0.10)"
          />

          <text
            x="38"
            y="583"
            fill="#a16207"
            fontSize="9"
            fontWeight="600"
            letterSpacing="1.3"
          >
            RELATED TRANSACTIONS
          </text>

          {/* Direct relationships */}
          {directEdges.map((edge, index) => {
            const source =
              getPosition(edge.source);

            const target =
              getPosition(edge.target);

            if (!source || !target) {
              return null;
            }

            return (
              <path
                key={`direct-${index}`}
                d={directPath(
                  source,
                  target,
                )}
                fill="none"
                stroke="rgba(100,116,139,0.38)"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            );
          })}

          {/* Shared relationships */}
          {sharedEdges.map((edge, index) => {
            const source =
              getPosition(edge.source);

            const target =
              getPosition(edge.target);

            if (!source || !target) {
              return null;
            }

            return (
              <path
                key={`shared-${index}`}
                d={sharedPath(
                  index,
                  edge,
                  source,
                  target,
                )}
                fill="none"
                stroke="rgba(245,158,11,0.40)"
                strokeWidth="1.15"
                strokeDasharray="5 6"
                strokeLinecap="round"
              />
            );
          })}

          {/* Core nodes */}
          {coreNodes.map((node) => {
            const position =
              getPosition(node.id);

            if (!position) {
              return null;
            }

            const isSelected =
              node.type ===
                "TRANSACTION" &&
              node.selected;

            return (
              <g key={node.id}>
                {isSelected && (
                  <circle
                    cx={position.x}
                    cy={position.y}
                    r="50"
                    fill="rgba(16,185,129,0.035)"
                    stroke="rgba(16,185,129,0.16)"
                    strokeWidth="1"
                  />
                )}

                <circle
                  cx={position.x}
                  cy={position.y}
                  r={
                    isSelected
                      ? 35
                      : 30
                  }
                  fill={
                    isSelected
                      ? "#08281f"
                      : "#111827"
                  }
                  stroke={
                    isSelected
                      ? "#10b981"
                      : "rgba(148,163,184,0.28)"
                  }
                  strokeWidth={
                    isSelected
                      ? 2
                      : 1.3
                  }
                  filter={
                    isSelected
                      ? "url(#riskTransactionGlow)"
                      : undefined
                  }
                />

                <circle
                  cx={position.x}
                  cy={position.y}
                  r="17"
                  fill={
                    isSelected
                      ? "rgba(16,185,129,0.10)"
                      : "rgba(148,163,184,0.07)"
                  }
                />

                <foreignObject
                  x={position.x - 9}
                  y={position.y - 9}
                  width="18"
                  height="18"
                >
                  <div
                    className={`flex h-[18px] w-[18px] items-center justify-center ${
                      isSelected
                        ? "text-emerald-300"
                        : "text-slate-400"
                    }`}
                  >
                    {networkNodeIcon(
                      node.type,
                    )}
                  </div>
                </foreignObject>

                <text
                  x={position.x}
                  y={
                    position.y +
                    (isSelected
                      ? 57
                      : 51)
                  }
                  textAnchor="middle"
                  fill={
                    isSelected
                      ? "#34d399"
                      : "#94a3b8"
                  }
                  fontSize={
                    expanded ? "10" : "9"
                  }
                  fontWeight="600"
                  letterSpacing="0.8"
                >
                  {isSelected
                    ? "SELECTED TRANSACTION"
                    : getLabel(
                        node.type,
                      )}
                </text>

                <text
                  x={position.x}
                  y={
                    position.y +
                    (isSelected
                      ? 72
                      : 66)
                  }
                  textAnchor="middle"
                  fill="#475569"
                  fontSize={
                    expanded ? "8.5" : "8"
                  }
                >
                  {getValue(node)}
                </text>
              </g>
            );
          })}

          {/* Related transactions */}
          {relatedPositions.map(
            ({ node, x, y }, index) => (
              <g
                key={node.id}
                role={
                  onSelectTransaction
                    ? "button"
                    : undefined
                }
                tabIndex={
                  onSelectTransaction
                    ? 0
                    : undefined
                }
                aria-label={
                  onSelectTransaction
                    ? `Investigate related transaction ${index + 1}`
                    : undefined
                }
                onClick={() =>
                  onSelectTransaction?.(
                    node.label,
                  )
                }
                onKeyDown={(event) => {
                  if (
                    onSelectTransaction &&
                    (event.key === "Enter" ||
                      event.key === " ")
                  ) {
                    event.preventDefault();
                    onSelectTransaction(
                      node.label,
                    );
                  }
                }}
                className={
                  onSelectTransaction
                    ? "cursor-pointer"
                    : undefined
                }
              >
                <circle
                  cx={x}
                  cy={y}
                  r={expanded ? 33 : 31}
                  fill="#151208"
                  stroke="rgba(245,158,11,0.62)"
                  strokeWidth="1.3"
                />

                <circle
                  cx={x}
                  cy={y}
                  r="11"
                  fill="rgba(245,158,11,0.06)"
                  stroke="rgba(245,158,11,0.32)"
                />

                <circle
                  cx={x}
                  cy={y}
                  r="4"
                  fill="#f59e0b"
                />

                <text
                  x={x}
                  y={y + 51}
                  textAnchor="middle"
                  fill="#d6a84b"
                  fontSize={
                    expanded ? "8.5" : "8"
                  }
                  fontWeight="600"
                >
                  RELATED TX {index + 1}
                </text>

                <text
                  x={x}
                  y={y + 65}
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize={
                    expanded ? "7.8" : "7.5"
                  }
                >
                  {getValue(node)}
                </text>

                {node.amount !== null && (
                  <text
                    x={x}
                    y={y + 80}
                    textAnchor="middle"
                    fill="#475569"
                    fontSize={
                      expanded ? "7.8" : "7.5"
                    }
                  >
                    {formatCurrency(
                      node.amount,
                      node.currency ??
                        "INR",
                    )}
                  </text>
                )}

                {node.event_time && (
                  <text
                    x={x}
                    y={y + 94}
                    textAnchor="middle"
                    fill="#334155"
                    fontSize="6.5"
                  >
                    {formatTime(
                      node.event_time,
                    )}
                  </text>
                )}

                {onSelectTransaction && (
                  <text
                    x={x}
                    y={y + 108}
                    textAnchor="middle"
                    fill="#78551a"
                    fontSize="6.5"
                  >
                    CLICK TO INVESTIGATE
                  </text>
                )}
              </g>
            ),
          )}

          {/* Legend */}
          <g
            transform={`translate(38 ${
              graphHeight - 45
            })`}
          >
            <line
              x1="0"
              y1="0"
              x2="25"
              y2="0"
              stroke="rgba(100,116,139,0.45)"
              strokeWidth="1.5"
            />

            <text
              x="33"
              y="3"
              fill="#64748b"
              fontSize="8"
            >
              Direct relationship
            </text>

            <line
              x1="170"
              y1="0"
              x2="195"
              y2="0"
              stroke="rgba(245,158,11,0.48)"
              strokeWidth="1.2"
              strokeDasharray="5 6"
            />

            <text
              x="203"
              y="3"
              fill="#64748b"
              fontSize="8"
            >
              Shared entity
            </text>
          </g>

          {sharedRelationships.length > 0 && (
            <text
              x="38"
              y={graphHeight - 21}
              fill="#78551a"
              fontSize="7.5"
            >
              Shared entity types:{" "}
              {sharedRelationships.join(
                " · ",
              )}
            </text>
          )}
        </svg>
      </div>
    </div>
  );
}


function NetworkModal({
  transaction,
  network,
  onSelectTransaction,
  onClose,
}: {
  transaction: DashboardTransaction;
  network: TransactionNetwork;
  onSelectTransaction: (
    transactionId: string,
  ) => void | Promise<void>;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener(
      "keydown",
      handleKeyDown,
    );

    return () => {
      document.removeEventListener(
        "keydown",
        handleKeyDown,
      );
    };
  }, [onClose]);

  const handleBackdropClick = (
    event: ReactMouseEvent<HTMLDivElement>,
  ) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm sm:p-6"
      onMouseDown={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-label="Risk network investigation"
    >
      <div className="flex max-h-[95vh] w-full max-w-[1450px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#080b10] shadow-2xl">
        {/* Modal header */}
        <div className="flex shrink-0 items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Network
                size={16}
                className="text-emerald-400"
              />

              <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-slate-200">
                Risk Network Investigation
              </h2>
            </div>

            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-slate-600">
              <span className="font-mono">
                {shortId(
                  transaction.transaction_id,
                  24,
                )}
              </span>

              <span>
                {network.nodes.length} nodes
              </span>

              <span>
                {network.edges.length} relationships
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 text-slate-500 transition hover:bg-white/5 hover:text-white"
            aria-label="Close network investigation"
          >
            <X size={15} />
          </button>
        </div>

        {/* Modal body */}
        <div className="min-h-0 flex-1 overflow-auto p-4 sm:p-5">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
            {/* Graph */}
            <div className="min-w-0">
              <NetworkGraph
                network={network}
                expanded
                onSelectTransaction={
                  onSelectTransaction
                }
              />

              <div className="mt-4 rounded-xl border border-white/[0.07] bg-[#0b0f15]">
                <div className="border-b border-white/[0.06] px-4 py-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                    Connected transactions
                  </p>

                  <p className="mt-1 text-[9px] text-slate-600">
                    Click a related transaction to investigate it
                  </p>
                </div>

                <div className="grid gap-2 p-3 sm:grid-cols-2 xl:grid-cols-4">
                  {network.nodes
                    .filter(
                      (node) =>
                        node.type ===
                        "RELATED_TRANSACTION",
                    )
                    .map((node, index) => (
                      <div
                        key={node.id}
                        role="button"
                        tabIndex={0}
                        onClick={() =>
                          onSelectTransaction(
                            node.label,
                          )
                        }
                        onKeyDown={(event) => {
                          if (
                            event.key === "Enter" ||
                            event.key === " "
                          ) {
                            event.preventDefault();
                            onSelectTransaction(
                              node.label,
                            );
                          }
                        }}
                        className="cursor-pointer rounded-lg border border-white/[0.06] bg-white/[0.015] p-3 text-left transition hover:border-amber-400/25 hover:bg-amber-400/[0.03]"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-amber-500/30 bg-amber-500/[0.06] text-[8px] font-semibold text-amber-400">
                            {index + 1}
                          </span>

                          <span className="text-[8px] text-amber-500/70">
                            RELATED
                          </span>
                        </div>

                        <p className="mt-2 truncate font-mono text-[9px] text-slate-400">
                          {node.label}
                        </p>

                        <div className="mt-2 flex items-center justify-between text-[9px]">
                          <span className="text-slate-600">
                            {node.event_time
                              ? formatTime(
                                  node.event_time,
                                )
                              : "Unknown time"}
                          </span>

                          <span className="font-medium text-slate-400">
                            {node.amount !== null
                              ? formatCurrency(
                                  node.amount,
                                  node.currency ??
                                    "INR",
                                )
                              : "—"}
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Investigation summary */}
            <div className="space-y-3">
              <div className="rounded-xl border border-white/10 bg-[#0b0f15] p-4">
                <p className="text-[9px] uppercase tracking-[0.14em] text-slate-600">
                  Selected transaction
                </p>

                <p className="mt-2 font-mono text-[10px] text-slate-400">
                  {transaction.transaction_id}
                </p>

                <p className="mt-3 text-2xl font-semibold text-white">
                  {formatCurrency(
                    Number(transaction.amount),
                    transaction.currency,
                  )}
                </p>

                <div className="mt-4 grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-white/[0.06] bg-white/[0.015] p-3">
                    <p className="text-[8px] text-slate-600">
                      Risk
                    </p>

                    <p className="mt-1 text-xs font-semibold">
                      {transaction.risk_level ??
                        "PENDING"}
                    </p>
                  </div>

                  <div className="rounded-lg border border-white/[0.06] bg-white/[0.015] p-3">
                    <p className="text-[8px] text-slate-600">
                      Decision
                    </p>

                    <p className="mt-1 text-xs font-semibold">
                      {transaction.decision ??
                        "—"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#0b0f15] p-4">
                <p className="text-[9px] uppercase tracking-[0.14em] text-slate-600">
                  Network exposure
                </p>

                <div className="mt-3 grid grid-cols-2 gap-2">
                  <ExposureMetric
                    label="Related transactions"
                    value={
                      network.nodes.filter(
                        (node) =>
                          node.type ===
                          "RELATED_TRANSACTION",
                      ).length
                    }
                    highlighted={
                      relatedNodesCount(network) > 0
                    }
                  />

                  <ExposureMetric
                    label="Shared entity types"
                    value={
                      new Set(
                        network.edges
                          .filter((edge) =>
                            edge.type.startsWith(
                              "SHARED_",
                            ),
                          )
                          .map((edge) =>
                            edge.type.replace(
                              "SHARED_",
                              "",
                            ),
                          ),
                      ).size
                    }
                    highlighted={
                      network.edges.some(
                        (edge) =>
                          edge.type.startsWith(
                            "SHARED_",
                          ),
                      )
                    }
                  />
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#0b0f15] p-4">
                <p className="text-[9px] uppercase tracking-[0.14em] text-slate-600">
                  Shared entity signals
                </p>

                <div className="mt-3 flex flex-wrap gap-2">
                  {Array.from(
                    new Set(
                      network.edges
                        .filter((edge) =>
                          edge.type.startsWith(
                            "SHARED_",
                          ),
                        )
                        .map((edge) =>
                          edge.type.replace(
                            "SHARED_",
                            "",
                          ),
                        ),
                    ),
                  ).map((relationship) => (
                    <span
                      key={relationship}
                      className="rounded-md border border-amber-400/20 bg-amber-400/5 px-2 py-1 text-[8px] font-medium uppercase text-amber-300"
                    >
                      {relationship}
                    </span>
                  ))}

                  {!network.edges.some(
                    (edge) =>
                      edge.type.startsWith(
                        "SHARED_",
                      ),
                  ) && (
                    <span className="text-[9px] text-slate-600">
                      No shared-entity signals.
                    </span>
                  )}
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#0b0f15] p-4">
                <p className="text-[9px] uppercase tracking-[0.14em] text-slate-600">
                  Graph legend
                </p>

                <div className="mt-3 space-y-3 text-[9px] text-slate-500">
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-emerald-400" />
                    Selected transaction
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-amber-400" />
                    Related transaction
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="h-px w-6 bg-slate-500" />
                    Direct relationship
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="w-6 border-t border-dashed border-amber-500" />
                    Shared entity
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function relatedNodesCount(
  network: TransactionNetwork,
) {
  return network.nodes.filter(
    (node) =>
      node.type === "RELATED_TRANSACTION",
  ).length;
}


function ExposureMetric({
  label,
  value,
  highlighted,
}: {
  label: string;
  value: number;
  highlighted: boolean;
}) {
  return (
    <div
      className={`rounded-lg border px-3 py-2 ${
        highlighted
          ? "border-amber-400/20 bg-amber-400/5"
          : "border-white/[0.06] bg-white/[0.015]"
      }`}
    >
      <p className="text-[9px] text-slate-600">
        {label}
      </p>

      <p
        className={`mt-0.5 text-sm font-semibold ${
          highlighted
            ? "text-amber-300"
            : "text-slate-500"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function DetailMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
      <p className="text-[11px] text-slate-500">
        {label}
      </p>

      <p className="mt-1 font-semibold">
        {value}
      </p>
    </div>
  );
}

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/[0.05] py-2 last:border-0">
      <span className="shrink-0 text-slate-500">
        {label}
      </span>

      <span className="break-all text-right font-mono text-slate-400">
        {value}
      </span>
    </div>
  );
}
