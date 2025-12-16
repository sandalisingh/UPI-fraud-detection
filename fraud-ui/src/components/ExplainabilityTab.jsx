import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const FIELDS = [
  "Transaction_ID","Timestamp","Amount","Transaction_Type","Channel",
  "Sender_ID","Receiver_ID","Device_ID","Geo_Jump","Network_Type",
  "Amount_Change_Ratio","Is_First_Time_Receiver","Sender_Account_Age",
  "Avg_Transaction_Value","Txn_Count_1h","Time_Since_Last_Txn"
];

const OPTIONS = {
  Transaction_Type: ["P2P", "P2M", "Bill_Pay", "Collect_Request"],
  Channel: ["QR_Scan", "Intent_Link", "Manual_VPA"],
  Network_Type: ["4G", "5G", "Public_WiFi"]
};

const NUMERIC_FIELDS = [
  "Amount","Amount_Change_Ratio","Sender_Account_Age",
  "Avg_Transaction_Value","Txn_Count_1h","Time_Since_Last_Txn"
];

// Initialize form with defaults
const initialForm = FIELDS.reduce((acc, f) => {
  acc[f] = OPTIONS[f] ? OPTIONS[f][0] : ""; // first option or empty string
  return acc;
}, {});

export default function ExplainabilityTab() {
  const [form, setForm] = useState(initialForm);
  const [explanation, setExplanation] = useState(null);

  const handleChange = (key, value) => {
    if (NUMERIC_FIELDS.includes(key)) value = Number(value);
    setForm({ ...form, [key]: value });
  };

  const submit = async () => {
    try {
      const res = await fetch("https://upi-fraud-detection-sizg.onrender.com/predict_V2", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      setExplanation(data);
    } catch (err) {
      console.error("Error submitting form:", err);
    }
  };

  return (
    <div className="space-y-3">
      {FIELDS.map((f) => {
        // Use Select for categorical fields
        if (OPTIONS[f]) {
          return (
            <Select key={f} value={form[f]} onValueChange={(v) => handleChange(f, v)}>
              <SelectTrigger>
                <SelectValue placeholder={f.replaceAll("_", " ")} />
              </SelectTrigger>
              <SelectContent>
                {OPTIONS[f].map((opt) => (
                  <SelectItem key={opt} value={opt}>{opt}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          );
        }

        // Otherwise use Input
        return (
          <Input
            key={f}
            placeholder={f.replaceAll("_", " ")}
            value={form[f]}
            onChange={(e) => handleChange(f, e.target.value)}
          />
        );
      })}

      <Button className="w-full" onClick={submit}>Analyze Transaction Risk</Button>

      {explanation && (
        <div className="mt-4 p-4 rounded-xl border bg-red-50">
          <h2 className="text-lg font-bold text-red-700">🚩 FRAUD ALERT</h2>
          <p>Transaction blocked as potential fraud.</p>
          <p className="mt-1">Predicted fraud type: <b>{explanation.fraud_type}</b></p>

          <div className="mt-3">
            <p className="font-semibold">REASONING:</p>
            <ul className="list-disc ml-6 text-sm">
              {explanation.reasons
                .split("\n")
                .map((r, i) => <li key={i}>{r.replace(/^•\s*/, "")}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}