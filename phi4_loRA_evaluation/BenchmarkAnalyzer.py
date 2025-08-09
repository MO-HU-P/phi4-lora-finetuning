import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['font.family'] = ['DejaVu Sans']
try:
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    japanese_fonts = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'MS Gothic']
    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font, 'DejaVu Sans']
            break
except:
    pass 

class BenchmarkAnalyzer:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = None
        self.tasks = None
        
    def load_data(self):
        try:
            print(f"ファイル読み込み中: {self.json_file_path}")
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.tasks = self.data.get("タスク別詳細", [])
            metadata = self.data.get("評価メタデータ", {})
            print(f"データ読み込み完了")
            print(f"   タスク数: {len(self.tasks)}")
            print(f"   評価実行日時: {metadata.get('評価実行日時', 'N/A')}")
            print(f"   成功率: {metadata.get('成功率', 'N/A')}")
            print(f"   平均処理時間: {metadata.get('処理時間', 'N/A')}")
            
            eval_models = metadata.get('評価モデル', {})
            if eval_models:
                print(f"評価エージェント:")
                for role, model in eval_models.items():
                    print(f"      {role}: {model}")
                    
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return False
        return True
    
    def analyze_support_rates(self):
        manager_counts = defaultdict(int)
        
        judge_counts = {
            'deepseek-r1': defaultdict(int),
            'qwen3': defaultdict(int), 
            'gemma3': defaultdict(int)
        }
        
        for task in self.tasks:
            final_decision = task.get("最終判断", "")
            manager_counts[final_decision] += 1
            judge_details = task.get("Judge詳細評価", {})
            
            for judge_name, judge_data in judge_details.items():
                if judge_name in judge_counts:
                    preferred = judge_data.get("優れた回答", "")
                    judge_counts[judge_name][preferred] += 1
        
        total_tasks = len(self.tasks)
        
        print("=" * 60)
        print("A/B支持率分析結果")
        print("=" * 60)
        print(f"総タスク数: {total_tasks}")
        print()
        
        print("Manager Agent最終判断:")
        print("-" * 40)
        for decision, count in sorted(manager_counts.items()):
            percentage = (count / total_tasks) * 100
            model_name = "Base model" if decision == "A" else "LoRA model" if decision == "B" else "引き分け"
            print(f"  {decision} ({model_name}): {count:3d}問 ({percentage:5.1f}%)")
        
        lora_wins = manager_counts.get('B', 0)
        base_wins = manager_counts.get('A', 0)
        if base_wins + lora_wins > 0:
            lora_improvement = (lora_wins / (base_wins + lora_wins)) * 100
            print(f"\n LoRA改善効果: {lora_improvement:.1f}% (Tie除く)")
        
        print("\n" + "=" * 60)
        print("各Judge Agent判断詳細:")
        print("=" * 60)
        
        judge_names = ['deepseek-r1', 'qwen3', 'gemma3']
        print(f"{'Judge Agent':<12} {'A(Base)':<10} {'B(LoRA)':<10} {'その他':<8} {'LoRA勝率':<10}")
        print("-" * 55)
        
        for judge_name in judge_names:
            if judge_name in judge_counts:
                counts = judge_counts[judge_name]
                judge_total = sum(counts.values())
                a_count = counts.get('A', 0)
                b_count = counts.get('B', 0)
                other_count = judge_total - a_count - b_count
                if a_count + b_count > 0:
                    lora_rate = (b_count / (a_count + b_count)) * 100
                else:
                    lora_rate = 0
                print(f"{judge_name:<12} {a_count:<10d} {b_count:<10d} {other_count:<8d} {lora_rate:<8.1f}%")
        return manager_counts, judge_counts
    
    def analyze_score_breakdown(self):
        criteria = ["簡潔さ", "核心理解", "正確性", "明快さ"]
        criteria_counts = {criterion: defaultdict(int) for criterion in criteria}
        
        for task in self.tasks:
            score_breakdown = task.get("スコア内訳", {})
            
            for criterion in criteria:
                if criterion in score_breakdown:
                    decision = score_breakdown[criterion]
                    criteria_counts[criterion][decision] += 1
        
        print("\n" + "=" * 60)
        print("Manager判断：評価観点別A/B分析")
        print("=" * 60)
        total_tasks = len(self.tasks)
        
        print(f"{'評価観点':<12} {'A(Base)':<8} {'B(LoRA)':<8} {'Tie':<6} {'LoRA勝率':<10} {'改善度':<8}")
        print("-" * 58)
        
        overall_summary = {
            'base_total': 0,
            'lora_total': 0,
            'tie_total': 0
        }
        
        for criterion, counts in criteria_counts.items():
            a_count = counts.get('A', 0)
            b_count = counts.get('B', 0) 
            tie_count = counts.get('Tie', 0)
            overall_summary['base_total'] += a_count
            overall_summary['lora_total'] += b_count
            overall_summary['tie_total'] += tie_count
            
            decided_total = a_count + b_count
            if decided_total > 0:
                lora_win_rate = (b_count / decided_total) * 100
                improvement = "✅" if lora_win_rate > 50 else "❌"
            else:
                lora_win_rate = 0
                improvement = "-"
            
            print(f"{criterion:<12} {a_count:<8d} {b_count:<8d} {tie_count:<6d} {lora_win_rate:<8.1f}% {improvement:<8}")
        
        print("-" * 58)
        total_decided = overall_summary['base_total'] + overall_summary['lora_total']
        if total_decided > 0:
            overall_lora_rate = (overall_summary['lora_total'] / total_decided) * 100
            overall_improvement = "✅" if overall_lora_rate > 50 else "❌"
        else:
            overall_lora_rate = 0
            overall_improvement = "-"
            
        print(f"{'全体平均':<12} {overall_summary['base_total']:<8d} {overall_summary['lora_total']:<8d} {overall_summary['tie_total']:<6d} {overall_lora_rate:<8.1f}% {overall_improvement:<8}")
        
        print(f"\n LoRAファインチューニング効果サマリー:")
        print(f"   • 全観点平均LoRA勝率: {overall_lora_rate:.1f}%")
        print(f"   • 最も改善した観点: ", end="")
        
        best_criterion = ""
        best_rate = 0
        for criterion, counts in criteria_counts.items():
            a_count = counts.get('A', 0)
            b_count = counts.get('B', 0)
            if a_count + b_count > 0:
                rate = (b_count / (a_count + b_count)) * 100
                if rate > best_rate:
                    best_rate = rate
                    best_criterion = criterion
        
        if best_criterion:
            print(f"{best_criterion} ({best_rate:.1f}%)")
        else:
            print("データ不足")
        
        return criteria_counts
    
    def create_visualizations(self, manager_counts, criteria_counts):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        decisions = list(manager_counts.keys())
        counts = list(manager_counts.values())
        total = sum(counts)
        
        ax1.pie(counts, labels=[f'{d}\n({c}/{total})' for d, c in zip(decisions, counts)], 
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Manager Agent 最終判断分布', fontsize=14, fontweight='bold')
        
        criteria = ["簡潔さ", "核心理解", "正確性", "明快さ"]
        a_counts = [criteria_counts[c].get('A', 0) for c in criteria]
        b_counts = [criteria_counts[c].get('B', 0) for c in criteria]
        tie_counts = [criteria_counts[c].get('Tie', 0) for c in criteria]
        
        x = np.arange(len(criteria))
        width = 0.6
        
        p1 = ax2.bar(x, a_counts, width, label='A (Base)', color='lightcoral')
        p2 = ax2.bar(x, b_counts, width, bottom=a_counts, label='B (LoRA)', color='skyblue')
        p3 = ax2.bar(x, tie_counts, width, bottom=np.array(a_counts) + np.array(b_counts), 
                    label='Tie', color='lightgray')
        
        ax2.set_xlabel('評価観点')
        ax2.set_ylabel('問題数')
        ax2.set_title('Manager判断：評価観点別A/B分布', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(criteria)
        ax2.legend()
        
        for i, (a, b, tie) in enumerate(zip(a_counts, b_counts, tie_counts)):
            if a > 0:
                ax2.text(i, a/2, str(a), ha='center', va='center', fontweight='bold')
            if b > 0:
                ax2.text(i, a + b/2, str(b), ha='center', va='center', fontweight='bold')
            if tie > 0:
                ax2.text(i, a + b + tie/2, str(tie), ha='center', va='center', fontweight='bold')
        
        win_rates_a = []
        win_rates_b = []
        
        for criterion in criteria:
            total_decided = criteria_counts[criterion].get('A', 0) + criteria_counts[criterion].get('B', 0)
            if total_decided > 0:
                rate_a = (criteria_counts[criterion].get('A', 0) / total_decided) * 100
                rate_b = (criteria_counts[criterion].get('B', 0) / total_decided) * 100
            else:
                rate_a = rate_b = 0
            win_rates_a.append(rate_a)
            win_rates_b.append(rate_b)
        
        x = np.arange(len(criteria))
        width = 0.35
        
        ax3.bar(x - width/2, win_rates_a, width, label='A (Base)', color='lightcoral')
        ax3.bar(x + width/2, win_rates_b, width, label='B (LoRA)', color='skyblue')
        
        ax3.set_xlabel('評価観点')
        ax3.set_ylabel('勝率 (%)')
        ax3.set_title('評価観点別勝率比較 (Base vs LoRA, Tie除く)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(criteria)
        ax3.legend()
        ax3.set_ylim(0, 100)
        
        for i, (rate_a, rate_b) in enumerate(zip(win_rates_a, win_rates_b)):
            ax3.text(i - width/2, rate_a + 2, f'{rate_a:.1f}%', ha='center', va='bottom')
            ax3.text(i + width/2, rate_b + 2, f'{rate_b:.1f}%', ha='center', va='bottom')
        
        ax4.axis('off')
        
        total_tasks = len(self.tasks)
        a_total = manager_counts.get('A', 0)
        b_total = manager_counts.get('B', 0)
        tie_total = manager_counts.get('Tie', 0)
        
        summary_text = f"""
総合サマリー (Manager判断)

総タスク数: {total_tasks}

A (Base): {a_total} ({a_total/total_tasks*100:.1f}%)
B (LoRA): {b_total} ({b_total/total_tasks*100:.1f}%)
Tie: {tie_total} ({tie_total/total_tasks*100:.1f}%)

評価観点別Base勝率:
・簡潔さ: {win_rates_a[0]:.1f}%
・核心理解: {win_rates_a[1]:.1f}%
・正確性: {win_rates_a[2]:.1f}%
・明快さ: {win_rates_a[3]:.1f}%

LoRA改善効果:
・簡潔さ: {win_rates_b[0]:.1f}%
・核心理解: {win_rates_b[1]:.1f}%
・正確性: {win_rates_b[2]:.1f}%
・明快さ: {win_rates_b[3]:.1f}%
        """

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        if not self.load_data():
            return
        
        print("LLMベンチマーク評価結果分析")
        print(f"分析対象: {len(self.tasks)}タスク")
        print(f"評価モデル: Base model (A) vs LoRA model (B)")
        print()
        
        manager_counts, judge_counts = self.analyze_support_rates()
        criteria_counts = self.analyze_score_breakdown()
        self.print_final_summary(manager_counts, criteria_counts)
        print(f"\n グラフ生成中...")
        self.create_visualizations(manager_counts, criteria_counts)
        print(f" 分析完了！グラフが'benchmark_analysis.png'として保存されました。")
        
    def print_final_summary(self, manager_counts, criteria_counts):
        print("\n" + "=" * 60)
        print(" 総合評価サマリー")
        print("=" * 60)
        
        total_tasks = len(self.tasks)
        base_wins = manager_counts.get('A', 0)
        lora_wins = manager_counts.get('B', 0)
        ties = manager_counts.get('Tie', 0)
        
        print(f" 全体結果 (Manager判断):")
        print(f"   Base model勝利: {base_wins}/{total_tasks} ({base_wins/total_tasks*100:.1f}%)")
        print(f"   LoRA model勝利: {lora_wins}/{total_tasks} ({lora_wins/total_tasks*100:.1f}%)")
        print(f"   引き分け:       {ties}/{total_tasks} ({ties/total_tasks*100:.1f}%)")

        if lora_wins > base_wins:
            print(f"\n 結論: LoRAファインチューニングは成功です！")
            improvement = ((lora_wins - base_wins) / total_tasks) * 100
            print(f"   LoRAはベースモデルより{improvement:.1f}%ポイント優秀")
        elif base_wins > lora_wins:
            print(f"\n  結論: LoRAファインチューニングの効果は限定的")
            decline = ((base_wins - lora_wins) / total_tasks) * 100
            print(f"   ベースモデルがLoRAより{decline:.1f}%ポイント優秀")
        else:
            print(f"\n 結論: LoRAとベースモデルは同等の性能")
        
        print(f"\n LoRAが特に優秀な観点:")
        strengths = []
        for criterion, counts in criteria_counts.items():
            a_count = counts.get('A', 0)
            b_count = counts.get('B', 0)
            if a_count + b_count > 0:
                lora_rate = (b_count / (a_count + b_count)) * 100
                if lora_rate > 50:
                    strengths.append((criterion, lora_rate))
        
        if strengths:
            strengths.sort(key=lambda x: x[1], reverse=True)
            for i, (criterion, rate) in enumerate(strengths, 1):
                print(f"   {i}. {criterion}: {rate:.1f}%")
        else:
            print("   該当なし（全観点でベースモデル優勢）")
        
        print(f"\n 推奨事項:")
        if lora_wins > base_wins:
            print(f"   LoRAモデルの本格運用を推奨")
            if strengths:
                print(f"   特に「{strengths[0][0]}」が重要なタスクで効果的")
        else:
            print(f"   LoRAの追加チューニングまたはベースモデル継続使用を検討")
            print(f"   ファインチューニングデータやハイパーパラメータの見直しが必要")

if __name__ == "__main__":
    analyzer = BenchmarkAnalyzer('./output/assessment_report_*.json')
    analyzer.run_analysis()
    