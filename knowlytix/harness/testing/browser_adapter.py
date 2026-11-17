# SPDX-License-Identifier: Apache-2.0
"""GMSH Browser/UI Adapter — Playwright-based UI testing for agent systems.

Provides a structured adapter for testing agentic systems that have web
UIs.  Follows the SUTAdapter interface for drop-in integration with
the benchmark orchestrator.

Requires ``playwright`` (optional dependency)::

    pip install playwright
    playwright install chromium

Usage::

    adapter = BrowserAdapter(BrowserConfig(
        base_url="http://localhost:3000",
        headless=True,
    ))
    trace = adapter.evaluate("Navigate to dashboard and check metrics", "")
    # or execute a structured UI script:
    result = adapter.execute_ui_script(script)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .adapters import SUTTrace


# ---------------------------------------------------------------------------
# UI step and script models
# ---------------------------------------------------------------------------

@dataclass
class UIStep:
    """A single UI interaction step.

    Attributes
    ----------
    step_id : str
        Unique identifier within the script.
    action : str
        Action type: ``"navigate"``, ``"click"``, ``"type"``,
        ``"wait"``, ``"screenshot"``, ``"select"``, ``"scroll"``,
        ``"assert_text"``, ``"assert_visible"``, ``"assert_not_visible"``.
    selector : str
        CSS or XPath selector for the target element.
    value : str
        Value for type/select actions, URL for navigate, text for
        assert_text, milliseconds for wait.
    timeout_ms : float
        Timeout for this step (0 = use default).
    assertion : str
        Optional assertion to check after the step.
        Format: ``"text_contains:expected"`` or ``"visible:selector"``.
    screenshot : bool
        Whether to capture a screenshot after this step.
    """

    step_id: str = ""
    action: str = ""
    selector: str = ""
    value: str = ""
    timeout_ms: float = 5000.0
    assertion: str = ""
    screenshot: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> UIStep:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class UIStepResult:
    """Result of executing a single UI step."""

    step_id: str = ""
    action: str = ""
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    screenshot_path: str = ""
    assertion_passed: bool = True
    assertion_detail: str = ""


@dataclass
class UIScript:
    """A sequence of UI interaction steps.

    Attributes
    ----------
    script_id : str
        Unique identifier.
    name : str
        Human-readable name.
    description : str
        What this script tests.
    steps : list[UIStep]
        Ordered list of UI steps.
    setup_url : str
        URL to navigate to before executing steps.
    teardown_steps : list[UIStep]
        Steps to execute after the main script (cleanup).
    metadata : dict
        Arbitrary metadata.
    """

    script_id: str = ""
    name: str = ""
    description: str = ""
    steps: list[UIStep] = field(default_factory=list)
    setup_url: str = ""
    teardown_steps: list[UIStep] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> UIScript:
        steps = [UIStep.from_dict(s) for s in d.get("steps", [])]
        teardown = [UIStep.from_dict(s) for s in d.get("teardown_steps", [])]
        return cls(
            script_id=d.get("script_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            steps=steps,
            setup_url=d.get("setup_url", ""),
            teardown_steps=teardown,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> UIScript:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "selector": s.selector,
                    "value": s.value,
                    "timeout_ms": s.timeout_ms,
                    "assertion": s.assertion,
                    "screenshot": s.screenshot,
                }
                for s in self.steps
            ],
            "setup_url": self.setup_url,
            "teardown_steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "selector": s.selector,
                    "value": s.value,
                }
                for s in self.teardown_steps
            ],
            "metadata": self.metadata,
        }


@dataclass
class UIScriptResult:
    """Aggregate result of executing a UI script."""

    script_id: str = ""
    step_results: list[UIStepResult] = field(default_factory=list)
    overall_passed: bool = False
    total_latency_ms: float = 0.0
    screenshots: list[str] = field(default_factory=list)
    page_text: str = ""

    def to_dict(self) -> dict:
        return {
            "script_id": self.script_id,
            "overall_passed": self.overall_passed,
            "n_steps": len(self.step_results),
            "steps_passed": sum(
                1 for s in self.step_results if s.success
            ),
            "total_latency_ms": self.total_latency_ms,
            "screenshots": self.screenshots,
            "step_details": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "success": s.success,
                    "error": s.error,
                    "latency_ms": s.latency_ms,
                    "assertion_passed": s.assertion_passed,
                }
                for s in self.step_results
            ],
        }


# ---------------------------------------------------------------------------
# Browser configuration
# ---------------------------------------------------------------------------

@dataclass
class BrowserConfig:
    """Configuration for the browser adapter.

    Attributes
    ----------
    base_url : str
        Base URL for the application under test.
    headless : bool
        Run browser in headless mode.
    browser_type : str
        Browser engine: ``"chromium"``, ``"firefox"``, ``"webkit"``.
    viewport_width : int
        Browser viewport width.
    viewport_height : int
        Browser viewport height.
    default_timeout_ms : float
        Default timeout for actions.
    screenshot_dir : str
        Directory to save screenshots.
    """

    base_url: str = ""
    headless: bool = True
    browser_type: str = "chromium"
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout_ms: float = 10000.0
    screenshot_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "headless": self.headless,
            "browser_type": self.browser_type,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "default_timeout_ms": self.default_timeout_ms,
            "screenshot_dir": self.screenshot_dir,
        }


# ---------------------------------------------------------------------------
# Browser adapter
# ---------------------------------------------------------------------------

class BrowserAdapter:
    """Playwright-based adapter for UI testing of agentic systems.

    Implements the SUTAdapter interface: ``evaluate(question, context)``
    navigates to the base URL, types the question, and returns the
    page's visible text as the response.

    Also provides ``execute_ui_script()`` for structured multi-step
    UI testing.

    Parameters
    ----------
    config : BrowserConfig
        Browser configuration.
    """

    def __init__(self, config: BrowserConfig):
        self.config = config
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_browser(self):
        """Lazily initialize Playwright and browser."""
        if self._page is not None:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for BrowserAdapter. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self._playwright = sync_playwright().start()

        launcher = getattr(self._playwright, self.config.browser_type)
        self._browser = launcher.launch(headless=self.config.headless)
        self._context = self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.config.default_timeout_ms)

    def close(self):
        """Close browser and Playwright."""
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None

    def __del__(self):
        self.close()

    # -- SUTAdapter interface ----------------------------------------------

    def evaluate(self, question: str, context: str) -> "SUTTrace":
        """Execute a question against the web UI.

        Navigates to base_url, finds an input field, types the question,
        and captures the response text from the page.
        """
        from .adapters import SUTTrace, ToolCallRecord

        self._ensure_browser()
        t0 = time.time()
        tool_records = []

        try:
            # Navigate to base URL
            if self.config.base_url:
                self._page.goto(self.config.base_url)
                tool_records.append(ToolCallRecord(
                    tool_name="navigate",
                    arguments={"url": self.config.base_url},
                    result="navigated",
                ))

            # Try to find an input field and submit the question
            input_sel = (
                "input[type='text'], textarea, "
                "[contenteditable='true'], [role='textbox']"
            )
            input_el = self._page.query_selector(input_sel)
            if input_el:
                input_el.fill(question)
                tool_records.append(ToolCallRecord(
                    tool_name="type",
                    arguments={"text": question[:100]},
                    result="typed",
                ))

                # Try to submit (Enter key or submit button)
                submit = self._page.query_selector(
                    "button[type='submit'], button:has-text('Submit'), "
                    "button:has-text('Send'), button:has-text('Ask')"
                )
                if submit:
                    submit.click()
                else:
                    input_el.press("Enter")

                # Wait for response
                self._page.wait_for_timeout(2000)

            # Capture page text as the response
            raw_output = self._page.inner_text("body")
            elapsed = (time.time() - t0) * 1000

            return SUTTrace(
                raw_output=raw_output[:5000],
                tool_calls=tool_records,
                latency_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.time() - t0) * 1000
            return SUTTrace(
                raw_output="",
                tool_calls=tool_records,
                latency_ms=elapsed,
                error=str(exc),
            )

    # -- Structured UI script execution ------------------------------------

    def execute_ui_script(self, script: UIScript) -> UIScriptResult:
        """Execute a structured UI test script.

        Parameters
        ----------
        script : UIScript
            The script to execute.

        Returns
        -------
        UIScriptResult
            Aggregate result with per-step details.
        """
        self._ensure_browser()
        result = UIScriptResult(script_id=script.script_id)
        t0_total = time.time()

        # Navigate to setup URL
        if script.setup_url:
            self._page.goto(script.setup_url)

        # Execute main steps
        for step in script.steps:
            step_result = self._execute_step(step)
            result.step_results.append(step_result)
            if step_result.screenshot_path:
                result.screenshots.append(step_result.screenshot_path)

        # Execute teardown steps (best-effort)
        for step in script.teardown_steps:
            self._execute_step(step)

        # Capture final page text
        try:
            result.page_text = self._page.inner_text("body")[:5000]
        except Exception:
            result.page_text = ""

        result.total_latency_ms = (time.time() - t0_total) * 1000
        result.overall_passed = all(
            s.success and s.assertion_passed
            for s in result.step_results
        )
        return result

    def _execute_step(self, step: UIStep) -> UIStepResult:
        """Execute a single UI step."""
        sr = UIStepResult(step_id=step.step_id, action=step.action)
        timeout = step.timeout_ms or self.config.default_timeout_ms
        t0 = time.time()

        try:
            if step.action == "navigate":
                url = step.value or step.selector
                self._page.goto(url, timeout=timeout)

            elif step.action == "click":
                self._page.click(step.selector, timeout=timeout)

            elif step.action == "type":
                self._page.fill(step.selector, step.value, timeout=timeout)

            elif step.action == "select":
                self._page.select_option(
                    step.selector, step.value, timeout=timeout,
                )

            elif step.action == "wait":
                wait_ms = float(step.value) if step.value else 1000
                self._page.wait_for_timeout(wait_ms)

            elif step.action == "scroll":
                self._page.evaluate(
                    f"window.scrollBy(0, {step.value or 500})"
                )

            elif step.action == "screenshot":
                path = step.value or f"screenshot_{step.step_id}.png"
                if self.config.screenshot_dir:
                    import os
                    path = os.path.join(self.config.screenshot_dir, path)
                self._page.screenshot(path=path)
                sr.screenshot_path = path

            elif step.action == "assert_text":
                text = self._page.inner_text(step.selector or "body")
                if step.value not in text:
                    sr.assertion_passed = False
                    sr.assertion_detail = (
                        f"Expected text '{step.value}' not found"
                    )

            elif step.action == "assert_visible":
                el = self._page.query_selector(step.selector)
                if el is None or not el.is_visible():
                    sr.assertion_passed = False
                    sr.assertion_detail = (
                        f"Element '{step.selector}' not visible"
                    )

            elif step.action == "assert_not_visible":
                el = self._page.query_selector(step.selector)
                if el is not None and el.is_visible():
                    sr.assertion_passed = False
                    sr.assertion_detail = (
                        f"Element '{step.selector}' unexpectedly visible"
                    )

            else:
                sr.success = False
                sr.error = f"Unknown action: {step.action}"

        except Exception as exc:
            sr.success = False
            sr.error = f"{type(exc).__name__}: {exc}"

        sr.latency_ms = (time.time() - t0) * 1000

        # Post-step screenshot if requested
        if step.screenshot and not sr.screenshot_path:
            try:
                path = f"screenshot_{step.step_id}.png"
                if self.config.screenshot_dir:
                    import os
                    path = os.path.join(self.config.screenshot_dir, path)
                self._page.screenshot(path=path)
                sr.screenshot_path = path
            except Exception:
                pass

        # Post-step assertion
        if step.assertion and sr.success:
            sr.assertion_passed, sr.assertion_detail = self._check_assertion(
                step.assertion,
            )

        return sr

    def _check_assertion(self, assertion: str) -> tuple[bool, str]:
        """Evaluate a step assertion string.

        Formats:
        - ``"text_contains:expected text"``
        - ``"visible:selector"``
        - ``"not_visible:selector"``
        - ``"url_contains:fragment"``
        """
        if ":" not in assertion:
            return False, f"Invalid assertion format: {assertion}"

        atype, value = assertion.split(":", 1)

        if atype == "text_contains":
            text = self._page.inner_text("body")
            if value in text:
                return True, f"Text contains '{value}'"
            return False, f"Text does not contain '{value}'"

        if atype == "visible":
            el = self._page.query_selector(value)
            if el and el.is_visible():
                return True, f"Element '{value}' is visible"
            return False, f"Element '{value}' not visible"

        if atype == "not_visible":
            el = self._page.query_selector(value)
            if el is None or not el.is_visible():
                return True, f"Element '{value}' not visible (expected)"
            return False, f"Element '{value}' unexpectedly visible"

        if atype == "url_contains":
            url = self._page.url
            if value in url:
                return True, f"URL contains '{value}'"
            return False, f"URL '{url}' does not contain '{value}'"

        return False, f"Unknown assertion type: {atype}"

    def __repr__(self) -> str:
        return f"BrowserAdapter(base_url={self.config.base_url!r})"
