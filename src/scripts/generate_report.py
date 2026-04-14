import glob
import json
import os
from datetime import datetime


def generate_html():
    results_dir = "deepeval_results"

    if not os.path.exists(results_dir):
        print(f"Error: Could not find {results_dir} directory.")
        return

    json_files = glob.glob(f"{results_dir}/*.json")
    if not json_files:
        print(f"Warning: No JSON files found in {results_dir}.")
        return

    print(f"Found {len(json_files)} test suites. Building tabbed report...")

    # 1. Process all the data first
    suites = {}
    for file_path in json_files:
        suite_id = os.path.basename(file_path).replace(".json", "")
        suite_name = suite_id.replace("_", " ").title()

        with open(file_path, "r") as f:
            data = json.load(f)

        test_cases = data.get("testRunData", {}).get("testCases", [])
        success_count = sum(1 for case in test_cases if case.get("success", False))
        failed_count = len(test_cases) - success_count
        total_count = len(test_cases)
        pass_rate = (success_count / total_count * 100) if total_count else 0

        suites[suite_id] = {
            "name": suite_name,
            "cases": test_cases,
            "summary": {
                "success": success_count,
                "failed": failed_count,
                "total": total_count,
                "pass_rate": pass_rate,
            },
        }

    # 2. Build the HTML Header & Navigation Tabs
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DeepEval CI Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            function openTab(evt, tabName) {{
                // Hide all panels
                document.querySelectorAll('.tab-panel').forEach(panel => {{
                    panel.classList.add('hidden');
                }});
                // Reset all buttons to inactive state
                document.querySelectorAll('.tab-button').forEach(btn => {{
                    btn.classList.remove('text-blue-600', 'border-blue-600');
                    btn.classList.add('text-gray-500', 'border-transparent');
                }});
                // Show target panel
                document.getElementById(tabName).classList.remove('hidden');
                // Set clicked button to active state
                evt.currentTarget.classList.remove('text-gray-500', 'border-transparent');
                evt.currentTarget.classList.add('text-blue-600', 'border-blue-600');
            }}
        </script>
    </head>
    <body class="bg-gray-50 p-8 text-gray-800">
        <div class="max-w-[1600px] mx-auto">
            <h1 class="text-3xl font-bold mb-2">DeepEval Evaluation Report</h1>
            <p class="text-gray-500 mb-6">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="mb-4 border-b border-gray-200">
                <ul class="flex flex-wrap -mb-px text-sm font-medium text-center">
    """

    # Generate the Tab Buttons
    is_first = True
    for suite_id, suite_data in suites.items():
        active_text = (
            "text-blue-600 border-blue-600"
            if is_first
            else "text-gray-500 border-transparent"
        )
        html_content += f"""
                    <li class="mr-2">
                        <button class="inline-block p-4 rounded-t-lg border-b-2 {active_text} hover:text-gray-600 hover:border-gray-300 tab-button"
                                onclick="openTab(event, '{suite_id}')">
                            {suite_data["name"]}
                        </button>
                    </li>
        """
        is_first = False

    html_content += """
                </ul>
            </div>

            <div id="tabContent">
    """

    # 3. Generate the Tables (One for each suite)
    is_first = True
    for suite_id, suite_data in suites.items():
        visibility = "" if is_first else "hidden"
        summary = suite_data["summary"]

        html_content += f"""
                <div class="{visibility} tab-panel" id="{suite_id}">
                    <div class="grid grid-cols-1 sm:grid-cols-4 gap-3 mb-4">
                        <div class="bg-green-50 border border-green-200 rounded-lg px-4 py-3">
                            <p class="text-xs font-semibold uppercase tracking-wide text-green-700">Success</p>
                            <p class="text-2xl font-bold text-green-900">{summary["success"]}</p>
                        </div>
                        <div class="bg-red-50 border border-red-200 rounded-lg px-4 py-3">
                            <p class="text-xs font-semibold uppercase tracking-wide text-red-700">Failed</p>
                            <p class="text-2xl font-bold text-red-900">{summary["failed"]}</p>
                        </div>
                        <div class="bg-slate-50 border border-slate-200 rounded-lg px-4 py-3">
                            <p class="text-xs font-semibold uppercase tracking-wide text-slate-600">Total</p>
                            <p class="text-2xl font-bold text-slate-800">{summary["total"]}</p>
                        </div>
                        <div class="bg-indigo-50 border border-indigo-200 rounded-lg px-4 py-3">
                            <p class="text-xs font-semibold uppercase tracking-wide text-indigo-700">Pass Rate</p>
                            <p class="text-2xl font-bold text-indigo-900">{summary["pass_rate"]:.1f}%</p>
                        </div>
                    </div>
                    <div class="bg-white shadow-md rounded-lg overflow-hidden">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-800 text-white">
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider w-24">Status</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider w-1/4">Input Prompt</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider w-1/3">Agent Output</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider w-1/3">Judge Reasoning & Score</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-200">
        """

        for case in suite_data["cases"]:
            success = case.get("success", False)
            status_color = (
                "bg-green-100 text-green-800" if success else "bg-red-100 text-red-800"
            )
            status_text = "PASSED" if success else "FAILED"

            input_text = str(case.get("input", "N/A")).replace("\n", "<br>")
            actual_output = str(case.get("actualOutput", "N/A")).replace("\n", "<br>")

            metrics_html = ""
            for metric in case.get("metricsData", []):
                m_name = metric.get("name", "Unknown Metric")
                m_score = metric.get("score", 0)
                m_reason = str(metric.get("reason", "No reason provided.")).replace(
                    "\n", "<br>"
                )

                metrics_html += f"""
                                <div class='mb-4'>
                                    <strong class='text-gray-900'>{m_name} (Score: {m_score})</strong><br>
                                    <span class='text-sm text-gray-600'>{m_reason}</span>
                                </div>
                """

            html_content += f"""
                                <tr class="hover:bg-gray-50 align-top">
                                    <td class="px-6 py-4 whitespace-nowrap">
                                        <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full {status_color}">
                                            {status_text}
                                        </span>
                                    </td>
                                    <td class="px-6 py-4 text-sm text-gray-900">{input_text}</td>
                                    <td class="px-6 py-4 text-sm text-gray-900">{actual_output}</td>
                                    <td class="px-6 py-4 text-sm text-gray-900">{metrics_html}</td>
                                </tr>
            """

        html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """
        is_first = False

    # 4. Close the HTML document
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    os.makedirs("gh-pages-build", exist_ok=True)
    with open("gh-pages-build/index.html", "w") as f:
        f.write(html_content)

    print("Successfully generated tabbed gh-pages-build/index.html")


if __name__ == "__main__":
    generate_html()
