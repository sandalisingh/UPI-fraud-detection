import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export default function App() {
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
    isFlaggedFraud: 0, // default 0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key, value) => {
    setForm({ ...form, [key]: value });
  };

  // Toggle fraud flag
  const toggleFlag = () => {
    setForm({ ...form, isFlaggedFraud: form.isFlaggedFraud === 0 ? 1 : 0 });
  };

  const submit = async () => {
    setLoading(true);
    setResult(null);

    const payload = {
      ...form,
      step: Number(form.step),
      amount: Number(form.amount),
      oldbalanceOrg: Number(form.oldbalanceOrg),
      newbalanceOrig: Number(form.newbalanceOrig),
      oldbalanceDest: Number(form.oldbalanceDest),
      newbalanceDest: Number(form.newbalanceDest),
      isFlaggedFraud: Number(form.isFlaggedFraud),
    };

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-6">
      <Card className="w-full max-w-2xl shadow-xl rounded-2xl">
        <CardContent className="p-6 space-y-4">
          <h1 className="text-2xl font-bold text-center">UPI Fraud Detection</h1>

          <div>
            <label className="block font-medium mb-1">Step (Hour)</label>
            <Input onChange={(e) => handleChange("step", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Transaction Type</label>
            <Select onValueChange={(v) => handleChange("type", v)} defaultValue="TRANSFER">
              <SelectTrigger><SelectValue placeholder="Select Transaction Type" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="TRANSFER">TRANSFER</SelectItem>
                <SelectItem value="CASH_IN">CASH_IN</SelectItem>
                <SelectItem value="CASH_OUT">CASH_OUT</SelectItem>
                <SelectItem value="PAYMENT">PAYMENT</SelectItem>
                <SelectItem value="DEBIT">DEBIT</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block font-medium mb-1">Amount</label>
            <Input onChange={(e) => handleChange("amount", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Sender UPI ID</label>
            <Input onChange={(e) => handleChange("nameOrig", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Sender Old Balance</label>
            <Input onChange={(e) => handleChange("oldbalanceOrg", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Sender New Balance</label>
            <Input onChange={(e) => handleChange("newbalanceOrig", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Receiver UPI ID</label>
            <Input onChange={(e) => handleChange("nameDest", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Receiver Old Balance</label>
            <Input onChange={(e) => handleChange("oldbalanceDest", e.target.value)} />
          </div>

          <div>
            <label className="block font-medium mb-1">Receiver New Balance</label>
            <Input onChange={(e) => handleChange("newbalanceDest", e.target.value)} />
          </div>

          <Button className="w-full mt-2" onClick={submit} disabled={loading}>
            {loading ? "Analyzing..." : "Predict Fraud"}
          </Button>

          {result && (
            <div className="mt-4 p-4 rounded-xl bg-white border space-y-2">
              <p className="font-semibold">
                Prediction: {result.is_fraud === 1 ? "Fraud" : "Legitimate"}
              </p>
              {result.prediction === 1 && (
                <p className="text-red-600 font-medium">Fraud Type: {result.fraud_type}</p>
              )}
              <pre className="text-sm whitespace-pre-wrap text-slate-700">{result.explanation}</pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
