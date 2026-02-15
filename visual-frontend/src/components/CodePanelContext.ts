import { createContext } from "react";

export interface CodePanelContext {
  onOpenCodePanel: (entityName: string) => void;
  openPanels: Set<string>;
}

export const CodePanelCtx = createContext<CodePanelContext | null>(null);
