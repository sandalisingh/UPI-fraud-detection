import { Card, CardContent } from "@/components/ui/card";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// ---------------- CONFIG ----------------
const TRANSACTION_FIELDS = [
  { label: "Transaction ID", key: "Transaction_ID" },
  { label: "Timestamp", key: "Timestamp" },
  { label: "Receiver UPI ID", key: "Receiver_ID" },
  { label: "Mark if first time receiver", key: "Is_First_Time_Receiver" },
  { label: "Amount (daily transaction limit is of 1 Lakh)", key: "Amount" },
  { label: "Transaction type", key: "Transaction_Type" },
  { label: "Channel", key: "Channel" },
  { label: "Network type", key: "Network_Type" },
];

const USER_FIELDS = [
  { label: "Sender UPI ID", key: "Sender_ID" },
  { label: "Device ID", key: "Device_ID" },
  { label: "Geographical jump (0-5000)", key: "Geo_Jump" },
  { label: "Account age (in days)", key: "Sender_Account_Age" },
  { label: "Average transaction value", key: "Avg_Transaction_Value" },
  { label: "Transaction Count for past 1 hour", key: "Txn_Count_1h" },
  { label: "Last Transaction Time", key: "Last_Txn_Timestamp" },
];

const OPTIONS = {
  Transaction_Type: [
    { label: "P2P", value: "P2P" },
    { label: "P2M", value: "P2M" },
    { label: "Bill Pay", value: "Bill_Pay" },
    { label: "Collect Request", value: "Collect_Request" },
  ],
  Channel: [
    { label: "QR Scan", value: "QR_Scan" },
    { label: "Intent Link", value: "Intent_Link" },
    { label: "Manual VPA", value: "Manual_VPA" },
  ],
  Network_Type: [
    { label: "4G", value: "4G" },
    { label: "5G", value: "5G" },
    { label: "Public WiFi", value: "Public_WiFi" },
  ],
};

const NUMERIC_FIELDS = [
  "Amount",
  "Sender_Account_Age",
  "Avg_Transaction_Value",
  "Geo_Jump",
  "Txn_Count_1h",
];

// ---------------- INITIAL STATE ----------------
const initialForm = [...TRANSACTION_FIELDS, ...USER_FIELDS].reduce((acc, f) => {
  acc[f.key] = OPTIONS[f.key] ? OPTIONS[f.key][0] : "";
  if (f.key === "Is_First_Time_Receiver") acc[f.key] = false;
  return acc;
}, {});

export default function App() {
  const [error, setError] = useState(null);
  const [form, setForm] = useState(initialForm);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key, value) => {
    if (NUMERIC_FIELDS.includes(key)) {
      value = value === "" ? "" : parseFloat(value);
    }
    setForm({ ...form, [key]: value });
  };

  const submit = async () => {
    setLoading(true);
    setError(null);

    try {
      const payload = { ...form };

      // --- Compute Time_Since_Last_Txn ---
      if (form.Last_Txn_Timestamp && form.Timestamp) {
        const last = new Date(form.Last_Txn_Timestamp);
        const current = new Date(form.Timestamp);

        if (last > current) {
          throw new Error("Last transaction time cannot be after current transaction time");
        }

        payload.Time_Since_Last_Txn = Math.max(
          Math.floor((current - last) / 1000),
          0
        );
      } else {
        payload.Time_Since_Last_Txn = 99999;
      }

      delete payload.Last_Txn_Timestamp;

      const res = await fetch("https://upi-fraud-detection-sizg.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Fraud analysis failed");
      }

      setPrediction(data);

    } catch (err) {
      console.error(err);
      setError(err.message || "Unexpected error occurred");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };


  const renderField = ({ label, key }) => {
    if (key === "Is_First_Time_Receiver") {
      return (
        <div key={key} className="flex items-center justify-between">
          <label className="text-sm font-medium">{label}</label>
          <button
            type="button"
            className={`w-10 h-10 rounded-full border flex items-center justify-center ${form[key] ? "bg-green-500 text-white" : "bg-white"
              }`}
            onClick={() => handleChange(key, !form[key])}
          >
            {form[key] ? "✓" : ""}
          </button>
        </div>
      );
    }

    if (OPTIONS[key]) {
      return (
        <div key={key} className="space-y-1">
          <label className="text-sm font-medium">{label}</label>
          <Select value={form[key]} onValueChange={(v) => handleChange(key, v)}>
            <SelectTrigger>
              <SelectValue placeholder={`Select ${label}`} />
            </SelectTrigger>
            <SelectContent>
              {OPTIONS[key].map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

        </div>
      );
    }

    return (
      <div key={key} className="space-y-1">
        <label className="text-sm font-medium">{label}</label>
        <Input
          type={
            key === "Timestamp" || key === "Last_Txn_Timestamp"
              ? "datetime-local"
              : NUMERIC_FIELDS.includes(key)
                ? "number"
                : "text"
          }
          step={NUMERIC_FIELDS.includes(key) ? "any" : undefined}
          required
          value={form[key]}
          onChange={(e) => handleChange(key, e.target.value)}
        />
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-6">
      <Card className="w-full max-w-4xl shadow-xl rounded-2xl">
        <CardContent className="p-6 space-y-6">
          <h1 className="text-2xl font-bold text-center">UPI Fraud Detection</h1>

          {/* USER INFO */}
          <section>
            <h2 className="text-lg font-semibold mb-2">User Information</h2>
            <div className="grid grid-cols-2 gap-4">
              {USER_FIELDS.map(renderField)}
            </div>
          </section>

          {/* TRANSACTION INFO */}
          <section>
            <h2 className="text-lg font-semibold mb-2">Transaction Information</h2>
            <div className="grid grid-cols-2 gap-4">
              {TRANSACTION_FIELDS.map(renderField)}
            </div>
          </section>

          <Button className="w-full mt-2" onClick={submit} disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Transaction Risk"}
          </Button>

          {error && (
            <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-300 text-red-700">
              <b>Error:</b> {error}
            </div>
          )}

          {!error && prediction && (() => {
            const isFraud = prediction.fraud_type?.toLowerCase() !== "legit";

            return (
              <div
                className={`mt-4 p-4 rounded-xl border ${isFraud
                  ? "bg-red-50 border-red-300"
                  : "bg-green-50 border-green-300"
                  }`}
              >
                <div className="flex items-center justify-between">
                  <h2
                    className={`text-lg font-bold ${isFraud ? "text-red-700" : "text-green-700"
                      }`}
                  >
                    {isFraud ? "🚩 FRAUD ALERT" : "LEGIT TRANSACTION"}
                  </h2>

                  {prediction.risk_percent !== undefined && (
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-semibold ${prediction.risk_percent > 70
                        ? "bg-red-200 text-red-800"
                        : prediction.risk_percent > 40
                          ? "bg-yellow-200 text-yellow-800"
                          : "bg-green-200 text-green-800"
                        }`}
                    >
                      Risk: {prediction.risk_percent}%
                    </span>
                  )}
                </div>

                <p className="mt-1">
                  {isFraud
                    ? "Transaction blocked as potential fraud."
                    : "Transaction is safe and approved."}
                </p>

                {prediction.fraud_type && (
                  <p className="mt-1">
                    Predicted type: <b>{prediction.fraud_type}</b>
                  </p>
                )}

                {prediction.reasons?.length > 0 && (
                  <div className="mt-3">
                    <p className="font-semibold">REASONING:</p>
                    <ul className="list-disc ml-6 text-sm">
                      {prediction.reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })()}

        </CardContent>
      </Card>
    </div>
  );
}