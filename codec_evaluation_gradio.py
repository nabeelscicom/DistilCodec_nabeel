import os
os.environ['GRADIO_TEMP_DIR'] = './tmp'

import gradio as gr
import os
import json
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np # For plotting

matplotlib.use('Agg') # Use a non-interactive backend for matplotlib

# 假设的文件夹路径 - 请替换为实际的文件夹路径
AUDIO_FOLDERS = [
    "./data/org_audios",
    "./data/gen_audios"
]

# 评分维度
SCORING_DIMENSIONS = ["Fidelity of BGM", "Fidelity of Speech"]
SCORING_DIMENSIONS_CN = ["BGM Fidelity", "Speech Fidelity"]

        
def get_speaker_id_info(ref_speaker_id_info_list):
    speaker_id_info_dict = {}
    for path in ref_speaker_id_info_list:
        speaker_id_info_list = glob.glob(os.path.join(path, '*.json'))
        for speaker_id_info_path in speaker_id_info_list:
            speaker_id_info_name = os.path.basename(speaker_id_info_path).split('.')[0]
            # print(f'speaker_id_info_name: {speaker_id_info_name}', flush=True)
            speaker_id_detail = json.load(open(speaker_id_info_path, 'r'))
            speaker_id_info_dict[speaker_id_info_name] = speaker_id_detail # ''.join(speaker_id_info_dict['text_audio_tokens'])
    
    return speaker_id_info_dict


def load_data_from_excel(path):
    data_category_dict = {}

    # 读取Excel文件
    df = pd.read_excel(path)

    # 将数据框转换为字典
    data_dict = df.to_dict(orient='records')

    # 打印结果
    for item in data_dict:
        key = str(item['id']) + '_' + str(item['speaker_ids'])
        data_category_dict[key] = {'infer_text': item['question'],
                                   '类别': item['类别'],
                                   '二级类别': item['二级类别']}

    return data_category_dict


# 创建示例音频文件夹和文件 (用于测试，如果实际文件夹不存在)
def create_dummy_audio_files():
    sample_audio_names = ["audio_sample1.wav", "common_sample.mp3", "another_one.wav"]
    for folder_path in AUDIO_FOLDERS:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created dummy folder: {folder_path}")
        for audio_name in sample_audio_names:
            full_path = os.path.join(folder_path, audio_name)
            if not os.path.exists(full_path):
                try:
                    import wave
                    sample_rate = 16000
                    duration = 0.1 # seconds
                    frequency = 440 # Hz
                    t = np.linspace(0, duration, int(sample_rate * duration), False)
                    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
                    audio_data_pcm = (audio_data * 32767).astype(np.int16)
                    with wave.open(full_path, 'w') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2) # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data_pcm.tobytes())
                    print(f"Created dummy audio file: {full_path}")
                except Exception as e:
                    print(f"Could not create dummy wav {full_path}: {e}. Creating placeholder text file.")
                    with open(full_path, 'w') as f:
                        f.write("This is a dummy audio file.")
                    print(f"Created placeholder file: {full_path} (not real audio)")

# create_dummy_audio_files() # Call this to create dummy files if you don't have them

# 存储评估结果的DataFrame
EVALUATION_EXCEL_FILE = "evaluation_results_multi_dimension.xlsx"

# 构建列名
columns = ["audio_name"]
for i in range(3): # 3个模型
    for dim in SCORING_DIMENSIONS:
        columns.append(f"model_{i+1}_{dim}")
columns.extend(["comments", "evaluator", "timestamp"])

if os.path.exists(EVALUATION_EXCEL_FILE):
    evaluation_df = pd.read_excel(EVALUATION_EXCEL_FILE)
    for col in columns:
        if col not in evaluation_df.columns:
            if "score" in col or any(dim in col for dim in SCORING_DIMENSIONS):
                evaluation_df[col] = pd.NA
            else:
                evaluation_df[col] = ""
    evaluation_df = evaluation_df[columns]
else:
    evaluation_df = pd.DataFrame(columns=columns)

def get_audio_files():
    if not AUDIO_FOLDERS or not os.path.exists(AUDIO_FOLDERS[0]):
        gr.Warning(f"基准音频文件夹未找到: {AUDIO_FOLDERS[0] if AUDIO_FOLDERS else '未配置'}")
        return []
    try:
        files = sorted([f for f in os.listdir(AUDIO_FOLDERS[0]) if f.lower().endswith(('.wav', '.mp3'))])
        if not files:
            gr.Info(f"在 {AUDIO_FOLDERS[0]} 中未找到音频文件。")
        return files
    except Exception as e:
        gr.Error(f"从 {AUDIO_FOLDERS[0]} 读取音频文件时出错: {e}")
        return []
    

# data_category_dict
def load_audio_for_evaluation(audio_name):
    if not audio_name:
        return [None, None, None, "请选择一个音频文件。"]
    audio_paths = []
    all_exist = True
    for i, folder in enumerate(AUDIO_FOLDERS):
        print(f'audio_name is:{audio_name}', flush=True)
        path = os.path.join(folder, audio_name)
        if os.path.exists(path):
            audio_paths.append(path)
        else:
            audio_paths.append(None)
            all_exist = False
            gr.Warning(f"音频文件 '{audio_name}' 在文件夹 {i+1} ({folder}) 中未找到。")
    status_message = f"已加载 '{audio_name}'。" if all_exist else f"已加载 '{audio_name}' 但部分文件缺失 (请查看警告信息)。"
    while len(audio_paths) < 2:
        audio_paths.append(None)
    
    # print(f'ref_audio_path: {ref_audio_path}', flush=True)
    return audio_paths[0], audio_paths[1], status_message

def save_evaluation(audio_name, *scores_and_details):
    global evaluation_df
    if not audio_name:
        return "错误: 未选择用于评估的音频文件。"

    num_scores = len(SCORING_DIMENSIONS) * 2
    scores = scores_and_details[:num_scores]
    comments = scores_and_details[num_scores]
    evaluator = scores_and_details[num_scores+1]

    if not evaluator:
        return "错误: 请输入评估者名称。"

    # 验证分数是否在有效范围内 (0-5)
    for score_idx, score_val in enumerate(scores):
        if score_val is None: # 如果允许不打分，可以跳过或赋默认值
             return f"错误: 第 {score_idx//len(SCORING_DIMENSIONS) + 1} 个模型的第 {score_idx%len(SCORING_DIMENSIONS) +1 } 个维度分数未填写。"
        try:
            score_val_num = float(score_val)
            if not (1 <= score_val_num <= 10):
                return f"错误: 分数 {score_val_num} 超出有效范围 (0-5)。请检查第 {score_idx//len(SCORING_DIMENSIONS) + 1} 个模型的第 {score_idx%len(SCORING_DIMENSIONS) +1 } 个维度分数。"
        except ValueError:
            return f"错误: 无效的分数输入 '{score_val}'。请输入数字。"


    new_record = {"audio_name": audio_name}
    score_idx = 0
    for i in range(2):
        for dim in SCORING_DIMENSIONS:
            new_record[f"model_{i+1}_{dim}"] = float(scores[score_idx]) #确保保存为数字
            score_idx += 1
    new_record["comments"] = comments
    new_record["evaluator"] = evaluator
    new_record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing_indices = evaluation_df.index[
        (evaluation_df["audio_name"] == audio_name) &
        (evaluation_df["evaluator"] == evaluator)
    ].tolist()

    if existing_indices:
        for col, value in new_record.items():
            evaluation_df.loc[existing_indices[0], col] = value
        save_message = f"'{audio_name}' 由 '{evaluator}' 的评估已更新。"
    else:
        new_record_df = pd.DataFrame([new_record])
        evaluation_df = pd.concat([evaluation_df, new_record_df], ignore_index=True)
        save_message = f"'{audio_name}' 由 '{evaluator}' 的评估已保存。"
    try:
        evaluation_df.to_excel(EVALUATION_EXCEL_FILE, index=False)
    except Exception as e:
        return f"保存到Excel时出错: {e}"
    return save_message

def calculate_average_scores():
    if evaluation_df.empty:
        return "无评估数据可计算平均分。", None, None

    avg_scores_summary_list = []
    model_dimension_avg = {}
    overall_model_avg = {}

    for i in range(2):
        model_name = f"Model {i+1}"
        model_scores_sum = 0
        model_scores_count = 0
        avg_scores_summary_list.append(f"--- {model_name} ({os.path.basename(os.path.normpath(AUDIO_FOLDERS[i])) if i < len(AUDIO_FOLDERS) else 'N/A'}) ---")

        for idx, dim in enumerate(SCORING_DIMENSIONS):
            col_name = f"model_{i+1}_{dim}"
            dim_cn = SCORING_DIMENSIONS_CN[idx]
            if col_name in evaluation_df.columns: # 确保列存在
                # 转换为数字并移除NA/无效值
                numeric_scores = pd.to_numeric(evaluation_df[col_name], errors='coerce').dropna()
                if not numeric_scores.empty:
                    avg_score = numeric_scores.mean()
                    avg_scores_summary_list.append(f"  {dim_cn}: {avg_score:.2f}")
                    model_scores_sum += avg_score
                    model_scores_count += 1
                    if dim_cn not in model_dimension_avg:
                        model_dimension_avg[dim_cn] = {}
                    model_dimension_avg[dim_cn][model_name] = avg_score
                else:
                    avg_scores_summary_list.append(f"  {dim_cn}: N/A (无有效分数)")
                    if dim_cn not in model_dimension_avg:
                        model_dimension_avg[dim_cn] = {}
                    model_dimension_avg[dim_cn][model_name] = np.nan # 使用nan以便绘图时正确处理
            else:
                avg_scores_summary_list.append(f"  {dim_cn}: N/A (列不存在)")
                if dim_cn not in model_dimension_avg:
                    model_dimension_avg[dim_cn] = {}
                model_dimension_avg[dim_cn][model_name] = np.nan

        if model_scores_count > 0:
            overall_avg = model_scores_sum / model_scores_count
            avg_scores_summary_list.append(f"  综合平均分: {overall_avg:.2f}")
            overall_model_avg[model_name] = overall_avg
        else:
            avg_scores_summary_list.append(f"  综合平均分: N/A")
            overall_model_avg[model_name] = np.nan

    avg_scores_text = "\n".join(avg_scores_summary_list)
    plot_path_by_dimension = "average_scores_by_dimension.png"
    dims_plot = list(model_dimension_avg.keys())

    if not dims_plot or all(all(np.isnan(score) for score in model_scores.values()) for model_scores in model_dimension_avg.values()):
        return avg_scores_text, None, None

    n_dims = len(dims_plot)
    n_models = 2
    bar_width = 0.25
    index = np.arange(n_dims)
    plt.figure(figsize=(12, 7))
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i in range(n_models):
        model_key = f"Model {i+1}"
        model_avgs = [model_dimension_avg[dim].get(model_key, np.nan) for dim in dims_plot]
        plt.bar(index + i * bar_width, model_avgs, bar_width, label=f"{model_key} ({os.path.basename(os.path.normpath(AUDIO_FOLDERS[i]))})", color=colors[i])

    plt.xlabel("评分维度", fontweight='bold')
    plt.ylabel("平均分", fontweight='bold')
    plt.title("各模型在不同维度上的平均分对比", fontweight='bold')
    plt.xticks(index + bar_width * (n_models-1)/2, dims_plot)
    plt.legend(title="模型")

    # 动态调整Y轴范围
    all_scores_flat = [score for dim_scores in model_dimension_avg.values() for score in dim_scores.values() if pd.notna(score)]
    if all_scores_flat:
        min_score_val = min(all_scores_flat) if all_scores_flat else 0
        max_score_val = max(all_scores_flat) if all_scores_flat else 10
        plt.ylim(max(0, min_score_val - 1), min(10, max_score_val + 1))
    else:
        plt.ylim(0,10)

    plt.tight_layout()
    plt.savefig(plot_path_by_dimension)
    plt.close()

    plot_path_overall = "overall_average_scores.png"
    overall_model_names = [name for name, score in overall_model_avg.items() if pd.notna(score)]
    overall_model_scores = [score for score in overall_model_avg.values() if pd.notna(score)]

    if overall_model_names:
        plt.figure(figsize=(8, 5))
        plt.bar(overall_model_names, overall_model_scores, color=colors[:len(overall_model_names)])
        plt.title("各模型综合平均分")
        plt.ylabel("综合平均分")
        if overall_model_scores:
            min_overall_score = min(overall_model_scores)
            max_overall_score = max(overall_model_scores)
            plt.ylim(max(0, min_overall_score - 1), min(10, max_overall_score + 1))
        else:
            plt.ylim(0,10)
        plt.tight_layout()
        plt.savefig(plot_path_overall)
        plt.close()
    else:
        plot_path_overall = None
    return avg_scores_text, plot_path_by_dimension, plot_path_overall

def export_results():
    global evaluation_df
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_evaluation_results_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(filename) as writer:
            evaluation_df_to_export = evaluation_df.copy()
            for col in ["comments", "evaluator"]:
                if col in evaluation_df_to_export.columns:
                     evaluation_df_to_export[col] = evaluation_df_to_export[col].fillna('')
            evaluation_df_to_export.to_excel(writer, sheet_name="Detailed Evaluations", index=False)

            if not evaluation_df.empty:
                avg_scores_data = {}
                overall_avg_data = {}
                for i in range(3):
                    model_name_folder = os.path.basename(os.path.normpath(AUDIO_FOLDERS[i])) if i < len(AUDIO_FOLDERS) else f'Model_{i+1}'
                    model_label = f"Model {i+1} ({model_name_folder})"
                    current_model_scores = []
                    for idx, dim in enumerate(SCORING_DIMENSIONS):
                        col_name = f"model_{i+1}_{dim}"
                        dim_cn = SCORING_DIMENSIONS_CN[idx]
                        if col_name in evaluation_df.columns:
                            numeric_scores = pd.to_numeric(evaluation_df[col_name], errors='coerce').dropna()
                            if not numeric_scores.empty:
                                current_model_scores.append(numeric_scores.mean())
                            else:
                                current_model_scores.append("N/A (无分数)")
                        else:
                            current_model_scores.append("N/A (列缺失)")
                    avg_scores_data[model_label] = current_model_scores
                    numeric_model_dim_averages = [s for s in current_model_scores if isinstance(s, (int, float))]
                    if numeric_model_dim_averages:
                        overall_avg_data[model_label] = [sum(numeric_model_dim_averages) / len(numeric_model_dim_averages)]
                    else:
                        overall_avg_data[model_label] = ["N/A"]

                stats_df_dims = pd.DataFrame.from_dict(avg_scores_data, orient='index', columns=SCORING_DIMENSIONS_CN)
                stats_df_dims.index.name = "Model"
                stats_df_dims.reset_index(inplace=True)
                stats_df_overall = pd.DataFrame.from_dict(overall_avg_data, orient='index', columns=["综合平均分"])
                stats_df_overall.index.name = "Model"
                stats_df_overall.reset_index(inplace=True)
                stats_df_dims.to_excel(writer, sheet_name="Average Scores by Dimension", index=False)
                stats_df_overall.to_excel(writer, sheet_name="Overall Average Scores", index=False)
            else:
                pd.DataFrame({"Message": ["无评估数据可生成统计信息。"]}).to_excel(writer, sheet_name="Average Scores", index=False)
        return gr.File.update(value=filename, visible=True)
    except Exception as e:
        gr.Error(f"导出Excel文件失败: {e}")
        return gr.File.update(value=None, visible=False)

# --- Gradio UI ---
with gr.Blocks(title="Codec SBS 评估工具 (多维度)") as demo:
    gr.Markdown("# DistilCodec Side-by-Side(SBS)")
    gr.Markdown("Select an audio file, listen to three model versions, and assign integer scores between 0-5 for each model's fidelity, stability, naturalness, and emotional expressiveness, while providing comments.")

    with gr.Row():
        audio_selector = gr.Dropdown(choices=get_audio_files(), label="Choose Audio", info="The audio files in the first model folder will be listed as the benchmark.")
        evaluator_name = gr.Textbox(label="User Name", placeholder="Input your user name")

    gr.Markdown("### Audio Player & Multi-dimensional Rating (Please enter a number between 0 and 5)")

    all_score_inputs = [] # <--- RENAMED from all_score_sliders

    for i in range(2):
        with gr.Accordion(f"Model{i+1} - Scoring Section", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    audio_player = gr.Audio(label=f"Model{i+1} Audio", type="filepath", interactive=False)
                with gr.Column(scale=3):
                    model_dimension_inputs = [] # <--- RENAMED from model_dimension_sliders
                    for dim_idx, dim_cn_name in enumerate(SCORING_DIMENSIONS_CN):
                        dim_en_name = SCORING_DIMENSIONS[dim_idx]
                        # CHANGED: gr.Slider to gr.Number
                        score_input_field = gr.Number(
                            label=f"{dim_cn_name} (M{i+1})",
                            minimum=1,
                            maximum=5,
                            step=1, # For integer steps with up/down arrows
                            value=5, # Default value
                            interactive=True,
                            precision=0 # Ensures integer input
                        )
                        model_dimension_inputs.append(score_input_field)
                    all_score_inputs.extend(model_dimension_inputs) # <--- Use new list name
            if i == 0: audio_output1 = audio_player
            elif i == 1: audio_output2 = audio_player

    comments = gr.Textbox(label="Comments", placeholder="", lines=3)

    with gr.Row():
        submit_btn = gr.Button("Submit for Evaluation", variant="primary")
        export_btn = gr.Button("Export all results to Excel")
        stats_btn = gr.Button("Display/Refresh Average Score Statistics")

    status_Lbl = gr.Label(label="State")

    with gr.Accordion("View statistical data and exported files", open=False):
        avg_scores_display = gr.Textbox(label="Details of Average Scores for Each Model", lines=10, interactive=False)
        stats_plot_dim = gr.Image(label="Comparison of average scores by dimension", type="filepath", interactive=False, visible=False)
        stats_plot_overall = gr.Image(label="Model Composite Average Score", type="filepath", interactive=False, visible=False)
        export_file_display = gr.File(label="Download Assessment Excel", interactive=False, visible=False)

    audio_selector.change(
        fn=load_audio_for_evaluation,
        inputs=audio_selector,
        outputs=[audio_output1, audio_output2, status_Lbl]
    )

    # UPDATED: Use all_score_inputs
    save_inputs_list = [audio_selector] + all_score_inputs + [comments, evaluator_name]

    submit_btn.click(
        fn=save_evaluation,
        inputs=save_inputs_list, # <--- Use updated list
        outputs=status_Lbl
    )

    export_btn.click(
        fn=export_results,
        outputs=export_file_display
    ).then(
        fn=lambda: gr.File.update(visible=True),
        inputs=None,
        outputs=export_file_display
    )

    def show_stats_and_plots_combined():
        avg_text, plot_path_dim, plot_path_overall = calculate_average_scores()
        updates = [avg_text]
        if plot_path_dim and os.path.exists(plot_path_dim):
            updates.append(gr.Image.update(value=plot_path_dim, visible=True))
        else:
            if plot_path_dim: gr.Warning(f"维度对比图文件 {plot_path_dim} 未找到但预期存在。")
            updates.append(gr.Image.update(visible=False))
        if plot_path_overall and os.path.exists(plot_path_overall):
            updates.append(gr.Image.update(value=plot_path_overall, visible=True))
        else:
            if plot_path_overall: gr.Warning(f"综合平均分图文件 {plot_path_overall} 未找到但预期存在。")
            updates.append(gr.Image.update(visible=False))
        return tuple(updates)

    stats_btn.click(
        fn=show_stats_and_plots_combined,
        outputs=[avg_scores_display, stats_plot_dim, stats_plot_overall]
    )

if __name__ == "__main__":
    for folder in AUDIO_FOLDERS:
        os.makedirs(folder, exist_ok=True)
    launch_config = {
        "server_name": "0.0.0.0",
        "server_port": 8898,
        "share": True
    }
    demo.launch(**launch_config)