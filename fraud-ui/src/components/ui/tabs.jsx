import { createContext, useContext, useState } from "react";

const TabsContext = createContext();

export function Tabs({ children, defaultValue }) {
  const [activeTab, setActiveTab] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      <div>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ children, className }) {
  return <div className={`flex ${className}`}>{children}</div>;
}

export function TabsTrigger({ value, children }) {
  const { activeTab, setActiveTab } = useContext(TabsContext);
  const isActive = activeTab === value;

  return (
    <button
      onClick={() => setActiveTab(value)}
      className={`px-4 py-2 font-medium border-b-2 ${
        isActive ? "border-blue-500 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-700"
      }`}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children }) {
  const { activeTab } = useContext(TabsContext);
  if (activeTab !== value) return null;
  return <div>{children}</div>;
}
