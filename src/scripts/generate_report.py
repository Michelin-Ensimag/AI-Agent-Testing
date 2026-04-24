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

    print(f"Found {len(json_files)} test suites. Building visual dashboard...")

    # 1. Process all the data first
    suites = {}
    global_total = 0
    global_success = 0

    # For Chart.js
    chart_suite_names = []
    chart_suite_scores = []

    for file_path in json_files:
        suite_id = os.path.basename(file_path).replace(".json", "")
        suite_name = suite_id.replace("_", " ").title()

        with open(file_path, "r") as f:
            data = json.load(f)

        test_cases = data.get("testRunData", {}).get("testCases", [])

        success_count = 0
        all_metric_scores = []

        for case in test_cases:
            if case.get("success", False):
                success_count += 1
            for metric in case.get("metricsData", []):
                all_metric_scores.append(metric.get("score", 0))

        failed_count = len(test_cases) - success_count
        total_count = len(test_cases)
        pass_rate = (success_count / total_count * 100) if total_count else 0

        # Calculate average score for this specific suite
        avg_score = (
            (sum(all_metric_scores) / len(all_metric_scores))
            if all_metric_scores
            else 0
        )

        global_total += total_count
        global_success += success_count

        chart_suite_names.append(suite_name)
        chart_suite_scores.append(round(avg_score, 2))

        suites[suite_id] = {
            "name": suite_name,
            "cases": test_cases,
            "summary": {
                "success": success_count,
                "failed": failed_count,
                "total": total_count,
                "pass_rate": pass_rate,
                "avg_score": avg_score,
            },
        }

    global_pass_rate = (global_success / global_total * 100) if global_total else 0
    global_avg_score = (
        (sum(chart_suite_scores) / len(chart_suite_scores)) if chart_suite_scores else 0
    )

    # 2. Build the HTML Header (Injecting Chart.js and Lucide Icons)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DeepEval AI Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <script>
            function openTab(evt, tabName) {{
                document.querySelectorAll('.tab-panel').forEach(panel => {{
                    panel.classList.add('hidden');
                }});
                document.querySelectorAll('.tab-button').forEach(btn => {{
                    btn.classList.remove('text-blue-600', 'border-blue-600', 'bg-blue-50');
                    btn.classList.add('text-gray-500', 'border-transparent', 'hover:bg-gray-100');
                }});
                document.getElementById(tabName).classList.remove('hidden');
                evt.currentTarget.classList.remove('text-gray-500', 'border-transparent', 'hover:bg-gray-100');
                evt.currentTarget.classList.add('text-blue-600', 'border-blue-600', 'bg-blue-50');
            }}
        </script>
    </head>
    <body class="bg-slate-100 p-6 text-slate-800 font-sans">
        <div class="max-w-[1600px] mx-auto">

            <div class="flex items-center justify-between mb-8 bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <div>
                    <h1 class="text-3xl font-extrabold text-slate-900 flex items-center gap-3">
                        <i data-lucide="bot" class="w-8 h-8 text-blue-600"></i>
                        AI Agent DeepEval Dashboard
                    </h1>
                    <p class="text-slate-500 mt-1 flex items-center gap-2">
                        <i data-lucide="clock" class="w-4 h-4"></i> Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </p>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200 flex items-center gap-4">
                    <div class="p-4 bg-blue-100 text-blue-600 rounded-lg"><i data-lucide="target" class="w-8 h-8"></i></div>
                    <div>
                        <p class="text-sm font-semibold text-slate-500 uppercase tracking-wider">Total Tests</p>
                        <p class="text-3xl font-bold text-slate-900">{global_total}</p>
                    </div>
                </div>
                <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200 flex items-center gap-4">
                    <div class="p-4 bg-green-100 text-green-600 rounded-lg"><i data-lucide="check-circle" class="w-8 h-8"></i></div>
                    <div>
                        <p class="text-sm font-semibold text-slate-500 uppercase tracking-wider">Global Pass Rate</p>
                        <p class="text-3xl font-bold text-slate-900">{global_pass_rate:.1f}%</p>
                    </div>
                </div>
                <div class="bg-white rounded-xl shadow-sm p-6 border border-slate-200 flex items-center gap-4">
                    <div class="p-4 bg-purple-100 text-purple-600 rounded-lg"><i data-lucide="brain-circuit" class="w-8 h-8"></i></div>
                    <div>
                        <p class="text-sm font-semibold text-slate-500 uppercase tracking-wider">Avg Judge Score</p>
                        <p class="text-3xl font-bold text-slate-900">{global_avg_score:.2f} <span class="text-base font-normal text-slate-500">/ 1.0</span></p>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <h2 class="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2"><i data-lucide="pie-chart" class="w-5 h-5"></i> Execution Status</h2>
                    <div class="relative h-64 w-full flex justify-center">
                        <canvas id="passFailChart"></canvas>
                    </div>
                </div>

                <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                    <h2 class="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2"><i data-lucide="bar-chart-3" class="w-5 h-5"></i> Capability Scores by Suite</h2>
                    <div class="relative h-64 w-full">
                        <canvas id="scoreChart"></canvas>
                    </div>
                </div>
            </div>

            <h2 class="text-2xl font-bold text-slate-800 mb-4 mt-12 flex items-center gap-2"><i data-lucide="file-search" class="w-6 h-6 text-blue-600"></i> Detailed Suite Results</h2>
            <div class="mb-6 bg-white rounded-lg shadow-sm border border-slate-200 p-2">
                <ul class="flex flex-wrap text-sm font-medium text-center">
    """

    # Generate the Tab Buttons
    is_first = True
    for suite_id, suite_data in suites.items():
        active_text = (
            "text-blue-600 border-blue-600 bg-blue-50"
            if is_first
            else "text-gray-500 border-transparent hover:bg-gray-100"
        )
        html_content += f"""
                    <li class="mr-2">
                        <button class="inline-block px-6 py-3 rounded-lg border-b-2 transition-all duration-200 {active_text} tab-button flex items-center gap-2"
                                onclick="openTab(event, '{suite_id}')">
                            <i data-lucide="folder-code" class="w-4 h-4"></i> {suite_data["name"]}
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

                    <div class="flex gap-4 mb-4">
                        <span class="px-3 py-1 bg-white border border-slate-200 rounded-full text-sm font-semibold text-slate-600 shadow-sm flex items-center gap-2"><i data-lucide="hash" class="w-4 h-4"></i> Total: {summary["total"]}</span>
                        <span class="px-3 py-1 bg-green-50 border border-green-200 rounded-full text-sm font-semibold text-green-700 shadow-sm flex items-center gap-2"><i data-lucide="check" class="w-4 h-4"></i> Passed: {summary["success"]}</span>
                        <span class="px-3 py-1 bg-red-50 border border-red-200 rounded-full text-sm font-semibold text-red-700 shadow-sm flex items-center gap-2"><i data-lucide="x" class="w-4 h-4"></i> Failed: {summary["failed"]}</span>
                    </div>

                    <div class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
                        <table class="min-w-full divide-y divide-slate-200">
                            <thead class="bg-slate-800 text-white">
                                <tr>
                                    <th class="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider w-32">Status</th>
                                    <th class="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider w-1/4">Input Prompt</th>
                                    <th class="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider w-1/3">Agent Output</th>
                                    <th class="px-6 py-4 text-left text-xs font-medium uppercase tracking-wider w-1/3">Judge Reasoning & Score</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-200">
        """

        for case in suite_data["cases"]:
            success = case.get("success", False)

            if success:
                status_color = "bg-green-100 text-green-800 border-green-200"
                status_icon = "check-circle-2"
                status_text = "PASSED"
            else:
                status_color = "bg-red-100 text-red-800 border-red-200"
                status_icon = "x-circle"
                status_text = "FAILED"

            input_text = str(case.get("input", "N/A")).replace("\n", "<br>")
            actual_output = str(case.get("actualOutput", "N/A")).replace("\n", "<br>")

            metrics_html = ""
            for metric in case.get("metricsData", []):
                m_name = metric.get("name", "Unknown Metric")
                m_score = metric.get("score", 0)
                m_reason = str(metric.get("reason", "No reason provided.")).replace(
                    "\n", "<br>"
                )

                # Visual score indicator (green for high, orange for med, red for low)
                score_color = (
                    "text-green-600"
                    if m_score >= 0.8
                    else "text-orange-500"
                    if m_score >= 0.5
                    else "text-red-600"
                )

                metrics_html += f"""
                                <div class='mb-5 p-4 bg-slate-50 border border-slate-100 rounded-lg'>
                                    <div class='flex justify-between items-center mb-2'>
                                        <strong class='text-slate-800 font-bold'>{m_name}</strong>
                                        <span class='font-black {score_color} bg-white px-2 py-1 rounded shadow-sm border border-slate-200'>{m_score}</span>
                                    </div>
                                    <span class='text-sm text-slate-600 leading-relaxed'>{m_reason}</span>
                                </div>
                """

            html_content += f"""
                                <tr class="hover:bg-slate-50 align-top transition-colors">
                                    <td class="px-6 py-6 whitespace-nowrap">
                                        <span class="px-3 py-1.5 inline-flex items-center gap-1.5 text-xs font-bold rounded-full border {status_color}">
                                            <i data-lucide="{status_icon}" class="w-4 h-4"></i> {status_text}
                                        </span>
                                    </td>
                                    <td class="px-6 py-6 text-sm text-slate-800 font-medium">{input_text}</td>
                                    <td class="px-6 py-6 text-sm text-slate-700 whitespace-pre-wrap font-mono text-xs bg-slate-50 m-2 p-3 rounded border border-slate-100">{actual_output}</td>
                                    <td class="px-6 py-6">{metrics_html}</td>
                                </tr>
            """

        html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """
        is_first = False

    # 4. Inject Javascript for Charts and Icons
    # json.dumps safely converts python lists to JS arrays
    html_content += f"""
            </div>
        </div>

        <script>
            // Initialize Lucide Icons
            lucide.createIcons();

            // Chart.js Configuration
            const passFailCtx = document.getElementById('passFailChart').getContext('2d');
            new Chart(passFailCtx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Passed', 'Failed'],
                    datasets: [{{
                        data: [{global_success}, {global_total - global_success}],
                        backgroundColor: ['#22c55e', '#ef4444'],
                        borderWidth: 0,
                        hoverOffset: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom' }}
                    }},
                    cutout: '70%'
                }}
            }});

            const scoreCtx = document.getElementById('scoreChart').getContext('2d');
            new Chart(scoreCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(chart_suite_names)},
                    datasets: [{{
                        label: 'Average Score',
                        data: {json.dumps(chart_suite_scores)},
                        backgroundColor: '#3b82f6',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: true, max: 1 }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    os.makedirs("gh-pages-build", exist_ok=True)
    with open("gh-pages-build/index.html", "w") as f:
        f.write(html_content)

    print("Successfully generated advanced dashboard gh-pages-build/index.html")


if __name__ == "__main__":
    generate_html()
