import PredictionTab from "./components/PredictionTab";
import ExplainabilityTab from "./components/ExplainabilityTab";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center p-6">
      <Card className="w-full max-w-3xl shadow-xl rounded-2xl">
        <CardContent className="p-6">
          <h1 className="text-2xl font-bold text-center mb-4">UPI Fraud Detection</h1>

          <Tabs defaultValue="model1">
            <TabsList className="grid grid-cols-2 mb-6">
              <TabsTrigger value="model1">Model V1</TabsTrigger>
              <TabsTrigger value="model2">Model V2</TabsTrigger>
            </TabsList>

            <TabsContent value="model1">
              <PredictionTab />
            </TabsContent>

            <TabsContent value="model2">
              <ExplainabilityTab />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}