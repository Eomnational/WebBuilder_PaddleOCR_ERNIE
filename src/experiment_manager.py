import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ExperimentManager:
    def __init__(self, base_results_dir):
        self.base_results_dir = base_results_dir
        self.log_file = os.path.join(base_results_dir, 'experiment_log.csv')
        
        # 初始化日志文件
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'sample_name', 'model_version', 
                'processing_time', 'avg_confidence', 
                'regions_detected', 'images_extracted', 'tables_extracted'
            ])
            df.to_csv(self.log_file, index=False)

    def log_experiment(self, sample_name, model_version, metrics):
        """
        记录实验数据到 CSV 文件。
        """
        new_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_name': sample_name,
            'model_version': model_version,
            'processing_time': metrics.get('processing_time', 0),
            'avg_confidence': metrics.get('avg_confidence', 0),
            'regions_detected': metrics.get('regions_detected', 0),
            'images_extracted': metrics.get('images_extracted', 0),
            'tables_extracted': metrics.get('tables_extracted', 0)
        }
        
        df = pd.DataFrame([new_record])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        print(f"实验数据已记录到: {self.log_file}")

    def plot_charts(self, sample_name, output_dir):
        """
        绘制对比图表并保存。
        """
        if not os.path.exists(self.log_file):
            print("日志文件不存在，无法绘图。")
            return

        df = pd.read_csv(self.log_file)
        
        # 过滤当前样本的数据
        sample_df = df[df['sample_name'] == sample_name]
        
        if sample_df.empty:
            print(f"没有找到样本 {sample_name} 的数据。")
            return

        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        
        # 1. 绘制置信度对比图
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_version', y='avg_confidence', data=sample_df, palette='viridis')
        plt.title(f'Average Confidence by Model Version ({sample_name})')
        plt.ylabel('Average Confidence')
        plt.xlabel('Model Version')
        plt.ylim(0, 1.0)
        plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'))
        plt.close()
        
        # 2. 绘制处理时间对比图
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_version', y='processing_time', data=sample_df, palette='magma')
        plt.title(f'Processing Time by Model Version ({sample_name})')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model Version')
        plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
        plt.close()
        
        print(f"对比图表已保存到: {output_dir}")
