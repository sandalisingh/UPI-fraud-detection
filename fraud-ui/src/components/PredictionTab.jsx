// ================= components/PredictionTab.jsx =================
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function PredictionTab() {
  const [form, setForm] = useState({
    step: "",
    type: "TRANSFER",
    amount: "",
    nameOrig: "",
    oldbalanceOrg: "",
    newbalanceOrig: "",
    nameDest: "",
    oldbalanceDest: "",
    newbalanceDest: "",
    isFlaggedFraud: 0,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (k, v) => setForm({ ...form, [k]: v });

  const submit = async () => {
    setLoading(true);
    const payload = {
      ...form,
      step: Number(form.step),
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
    };

    const res = await fetch("http://127.0.0.1:8000/predict_V1", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    setResult(await res.json());
    setLoading(false);
  };

  return (
    <div className="space-y-3">
      <Input placeholder="Step (Hour)" onChange={(e) => handleChange("step", e.target.value)} />

      <Select defaultValue="TRANSFER" onValueChange={(v) => handleChange("type", v)}>
        <SelectTrigger><SelectValue placeholder="Transaction Type" /></SelectTrigger>
        <SelectContent>
          <SelectItem value="TRANSFER">TRANSFER</SelectItem>
          <SelectItem value="CASH_IN">CASH_IN</SelectItem>
          <SelectItem value="CASH_OUT">CASH_OUT</SelectItem>
          <SelectItem value="PAYMENT">PAYMENT</SelectItem>
          <SelectItem value="DEBIT">DEBIT</SelectItem>
        </SelectContent>
      </Select>

      <Input placeholder="Amount" onChange={(e) => handleChange("amount", e.target.value)} />
      <Input placeholder="Sender UPI ID" onChange={(e) => handleChange("nameOrig", e.target.value)} />
      <Input placeholder="Sender Old Balance" onChange={(e) => handleChange("oldbalanceOrg", e.target.value)} />
      <Input placeholder="Sender New Balance" onChange={(e) => handleChange("newbalanceOrig", e.target.value)} />
      <Input placeholder="Receiver UPI ID" onChange={(e) => handleChange("nameDest", e.target.value)} />
      <Input placeholder="Receiver Old Balance" onChange={(e) => handleChange("oldbalanceDest", e.target.value)} />
      <Input placeholder="Receiver New Balance" onChange={(e) => handleChange("newbalanceDest", e.target.value)} />

      <Button className="w-full" onClick={submit} disabled={loading}>
        {loading ? "Analyzing..." : "Predict Fraud"}
      </Button>

      {result && (
        <div className="mt-4 p-4 rounded-xl border bg-white">
          <p className="font-semibold">Prediction: {result.is_fraud ? "Fraud" : "Legitimate"}</p>
          {result.is_fraud === 1 && (
            <p className="text-red-600">Fraud Type: {result.fraud_type}</p>
          )}
          <pre className="text-sm mt-2 whitespace-pre-wrap">{result.explanation}</pre>
        </div>
      )}
    </div>
  );
}
