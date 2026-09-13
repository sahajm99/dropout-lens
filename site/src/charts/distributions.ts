import { showError } from "../figure.ts";

/**
 * Placeholder until the chart module lands (Tasks 8 to 10). The figure
 * reports itself as not built rather than failing silently, so the build
 * passes and the page says what is missing.
 */
export async function render(container: HTMLElement): Promise<void> {
  showError(container, "This figure is not built yet.", () => void render(container));
}
