import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# ============================================================================
# FOLDER MANAGEMENT FUNCTIONS
# ============================================================================

def create_output_folders():
    """Create organized folder structure for outputs"""
    
    # Get current timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    main_output_dir = f"advanced_markov_straddle_{timestamp}"
    
    # Create subdirectories
    folders = {
        'main': main_output_dir,
        'analysis': os.path.join(main_output_dir, 'markov_analysis'),
        'transitions': os.path.join(main_output_dir, 'transition_matrices'),
        'sequences': os.path.join(main_output_dir, 'outcome_sequences'),
        'quartiles': os.path.join(main_output_dir, 'quartile_analysis'),
        'drawdown': os.path.join(main_output_dir, 'drawdown_analysis')
    }
    
    # Create all directories
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
    
    print(f"📁 Created output folder structure: {main_output_dir}/")
    print(f"   📊 Markov analysis: {folders['analysis']}")
    print(f"   🔄 Transition matrices: {folders['transitions']}")
    print(f"   📈 Outcome sequences: {folders['sequences']}")
    print(f"   📊 Quartile analysis: {folders['quartiles']}")
    print(f"   📉 Drawdown analysis: {folders['drawdown']}")
    
    return folders

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================

def load_early_exit_results(summary_file_path):
    """Load early exit straddle results from previous analysis"""
    
    print(f"📊 Loading early exit results from: {summary_file_path}")
    
    try:
        # Read the summary CSV file
        df = pd.read_csv(summary_file_path)
        
        # Ensure we have required columns
        required_cols = ['asset', 'period', 'start_date', 'final_return_pct', 'early_exit', 'days_held', 'max_return_pct', 'min_return_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Convert dates
        df['start_date'] = pd.to_datetime(df['start_date'])
        
        # Sort by asset and start date to get chronological sequence
        df = df.sort_values(['asset', 'start_date'])
        
        print(f"✅ Loaded {len(df)} straddle outcomes for {df['asset'].nunique()} assets")
        print(f"📅 Date range: {df['start_date'].min().strftime('%Y-%m-%d')} to {df['start_date'].max().strftime('%Y-%m-%d')}")
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: Could not find {summary_file_path}")
        print("   Please run B_02_straddle_early_exit.py first to generate the data")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def define_performance_states(df):
    """Define sophisticated performance states based on returns and drawdowns"""
    
    print(f"\n🎯 Defining advanced performance states...")
    
    # Calculate overall quartiles for profitable trades
    profitable_trades = df[df['final_return_pct'] > 0]
    loss_trades = df[df['final_return_pct'] <= 0]
    
    if len(profitable_trades) == 0:
        print("❌ No profitable trades found")
        return None, None
    
    # Define quartiles for profitable trades
    profit_quartiles = profitable_trades['final_return_pct'].quantile([0.25, 0.5, 0.75]).tolist()
    
    print(f"📊 Profitable trade quartiles:")
    print(f"   Q1 (Small Win): 0% to {profit_quartiles[0]:.1f}%")
    print(f"   Q2 (Medium Win): {profit_quartiles[0]:.1f}% to {profit_quartiles[1]:.1f}%")
    print(f"   Q3 (Good Win): {profit_quartiles[1]:.1f}% to {profit_quartiles[2]:.1f}%")
    print(f"   Q4 (Big Win): {profit_quartiles[2]:.1f}%+")
    
    # Analyze losses - differentiate between regular losses and big drawdowns
    if len(loss_trades) > 0:
        # Define big drawdown threshold (e.g., worst 25% of losses)
        loss_threshold = loss_trades['final_return_pct'].quantile(0.25)  # Worst 25%
        print(f"   Big Drawdown: {loss_threshold:.1f}% or worse")
        print(f"   Regular Loss: {loss_threshold:.1f}% to 0%")
    else:
        loss_threshold = -10.0  # Default threshold
    
    # Create state classification function
    def classify_state(row):
        return_pct = row['final_return_pct']
        max_return = row['max_return_pct']
        min_return = row['min_return_pct']
        days_held = row['days_held']
        
        # Calculate drawdown during trade (unrealized)
        max_drawdown = max_return - min_return if pd.notna(max_return) and pd.notna(min_return) else 0
        
        if return_pct <= 0:
            # Loss states
            if return_pct <= loss_threshold:
                return 'BIG_DRAWDOWN'
            else:
                return 'REGULAR_LOSS'
        else:
            # Profitable states - consider both return and speed
            is_quick_win = days_held <= 90  # Quick if closed within 90 days
            
            if return_pct <= profit_quartiles[0]:
                return 'SMALL_WIN_QUICK' if is_quick_win else 'SMALL_WIN_SLOW'
            elif return_pct <= profit_quartiles[1]:
                return 'MEDIUM_WIN_QUICK' if is_quick_win else 'MEDIUM_WIN_SLOW'
            elif return_pct <= profit_quartiles[2]:
                return 'GOOD_WIN_QUICK' if is_quick_win else 'GOOD_WIN_SLOW'
            else:
                return 'BIG_WIN_QUICK' if is_quick_win else 'BIG_WIN_SLOW'
    
    # Apply classification
    df['performance_state'] = df.apply(classify_state, axis=1)
    
    # Display state distribution
    state_counts = df['performance_state'].value_counts()
    print(f"\n📈 Performance state distribution:")
    for state, count in state_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {state}: {count} ({pct:.1f}%)")
    
    return df, {
        'profit_quartiles': profit_quartiles,
        'loss_threshold': loss_threshold,
        'state_counts': state_counts
    }

def create_advanced_outcome_sequences(df):
    """Create sophisticated outcome sequences for each asset"""
    
    print(f"\n🔄 Creating advanced outcome sequences...")
    
    asset_sequences = {}
    
    for asset in df['asset'].unique():
        asset_data = df[df['asset'] == asset].copy()
        asset_data = asset_data.sort_values('start_date')
        
        # Create state sequence
        states = asset_data['performance_state'].tolist()
        returns = asset_data['final_return_pct'].tolist()
        days_held = asset_data['days_held'].tolist()
        dates = asset_data['start_date'].tolist()
        periods = asset_data['period'].tolist()
        max_returns = asset_data['max_return_pct'].tolist()
        min_returns = asset_data['min_return_pct'].tolist()
        
        asset_sequences[asset] = {
            'states': states,
            'returns': returns,
            'days_held': days_held,
            'dates': dates,
            'periods': periods,
            'max_returns': max_returns,
            'min_returns': min_returns,
            'total_trades': len(states),
            'state_distribution': pd.Series(states).value_counts().to_dict()
        }
        
        print(f"  {asset:>6}: {len(states)} trades, states: {len(set(states))}")
    
    print(f"✅ Created advanced sequences for {len(asset_sequences)} assets")
    
    return asset_sequences

# ============================================================================
# ADVANCED MARKOV ANALYSIS FUNCTIONS
# ============================================================================

def calculate_advanced_transition_probabilities(states):
    """Calculate transition probabilities between performance states"""
    
    if len(states) < 2:
        return None
    
    # Get unique states
    unique_states = list(set(states))
    n_states = len(unique_states)
    
    # Initialize transition matrix
    transition_counts = {from_state: {to_state: 0 for to_state in unique_states} 
                        for from_state in unique_states}
    
    # Count transitions
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        transition_counts[from_state][to_state] += 1
    
    # Calculate probabilities
    transition_probs = {}
    for from_state in unique_states:
        total_from = sum(transition_counts[from_state].values())
        if total_from > 0:
            transition_probs[from_state] = {
                to_state: count / total_from 
                for to_state, count in transition_counts[from_state].items()
            }
        else:
            transition_probs[from_state] = {to_state: 0 for to_state in unique_states}
    
    return {
        'transition_counts': transition_counts,
        'transition_probabilities': transition_probs,
        'unique_states': unique_states,
        'total_transitions': len(states) - 1
    }

def analyze_drawdown_recovery_patterns(asset_sequences):
    """Analyze recovery patterns after big drawdowns"""
    
    print(f"\n📉 Analyzing drawdown recovery patterns...")
    
    recovery_analysis = {}
    
    for asset, sequence_data in asset_sequences.items():
        states = sequence_data['states']
        returns = sequence_data['returns']
        days_held = sequence_data['days_held']
        
        if len(states) < 2:
            continue
        
        # Find all big drawdown events
        drawdown_indices = [i for i, state in enumerate(states) if state == 'BIG_DRAWDOWN']
        
        if len(drawdown_indices) == 0:
            continue
        
        recovery_patterns = []
        
        for dd_idx in drawdown_indices:
            if dd_idx < len(states) - 1:  # Must have a next trade
                next_state = states[dd_idx + 1]
                next_return = returns[dd_idx + 1]
                next_days = days_held[dd_idx + 1]
                drawdown_return = returns[dd_idx]
                
                # Classify recovery
                is_quick_recovery = next_days <= 90
                is_strong_recovery = next_return >= 20.0  # 20%+ return
                is_any_recovery = next_return > 0
                
                recovery_patterns.append({
                    'drawdown_return': drawdown_return,
                    'next_state': next_state,
                    'next_return': next_return,
                    'next_days': next_days,
                    'is_quick_recovery': is_quick_recovery,
                    'is_strong_recovery': is_strong_recovery,
                    'is_any_recovery': is_any_recovery
                })
        
        if len(recovery_patterns) > 0:
            recovery_analysis[asset] = {
                'total_drawdowns': len(recovery_patterns),
                'recovery_patterns': recovery_patterns,
                'recovery_rate': sum(p['is_any_recovery'] for p in recovery_patterns) / len(recovery_patterns),
                'quick_recovery_rate': sum(p['is_quick_recovery'] and p['is_any_recovery'] for p in recovery_patterns) / len(recovery_patterns),
                'strong_recovery_rate': sum(p['is_strong_recovery'] for p in recovery_patterns) / len(recovery_patterns),
                'avg_recovery_return': np.mean([p['next_return'] for p in recovery_patterns if p['is_any_recovery']]) if any(p['is_any_recovery'] for p in recovery_patterns) else 0,
                'avg_recovery_days': np.mean([p['next_days'] for p in recovery_patterns if p['is_any_recovery']]) if any(p['is_any_recovery'] for p in recovery_patterns) else 0
            }
            
            print(f"  {asset:>6}: {len(recovery_patterns)} drawdowns, {recovery_analysis[asset]['recovery_rate']*100:.0f}% recovery rate")
    
    print(f"✅ Analyzed recovery patterns for {len(recovery_analysis)} assets")
    
    return recovery_analysis

def analyze_win_quality_transitions(asset_sequences):
    """Analyze how win quality affects subsequent performance"""
    
    print(f"\n🏆 Analyzing win quality transition patterns...")
    
    win_analysis = {}
    
    # Define win state groups
    small_wins = ['SMALL_WIN_QUICK', 'SMALL_WIN_SLOW']
    medium_wins = ['MEDIUM_WIN_QUICK', 'MEDIUM_WIN_SLOW']
    good_wins = ['GOOD_WIN_QUICK', 'GOOD_WIN_SLOW']
    big_wins = ['BIG_WIN_QUICK', 'BIG_WIN_SLOW']
    
    win_groups = {
        'SMALL_WINS': small_wins,
        'MEDIUM_WINS': medium_wins,
        'GOOD_WINS': good_wins,
        'BIG_WINS': big_wins
    }
    
    for asset, sequence_data in asset_sequences.items():
        states = sequence_data['states']
        returns = sequence_data['returns']
        days_held = sequence_data['days_held']
        
        if len(states) < 2:
            continue
        
        win_transitions = {}
        
        for win_type, win_states in win_groups.items():
            transitions_after_win = []
            
            for i in range(len(states) - 1):
                if states[i] in win_states:
                    next_state = states[i + 1]
                    next_return = returns[i + 1]
                    next_days = days_held[i + 1]
                    current_return = returns[i]
                    current_days = days_held[i]
                    
                    transitions_after_win.append({
                        'current_return': current_return,
                        'current_days': current_days,
                        'next_state': next_state,
                        'next_return': next_return,
                        'next_days': next_days,
                        'next_is_profitable': next_return > 0,
                        'next_is_quick': next_days <= 90,
                        'next_is_big_win': next_return >= 20.0
                    })
            
            if len(transitions_after_win) > 0:
                win_transitions[win_type] = {
                    'count': len(transitions_after_win),
                    'next_profit_rate': sum(t['next_is_profitable'] for t in transitions_after_win) / len(transitions_after_win),
                    'next_quick_rate': sum(t['next_is_quick'] for t in transitions_after_win) / len(transitions_after_win),
                    'next_big_win_rate': sum(t['next_is_big_win'] for t in transitions_after_win) / len(transitions_after_win),
                    'avg_next_return': np.mean([t['next_return'] for t in transitions_after_win]),
                    'avg_next_days': np.mean([t['next_days'] for t in transitions_after_win]),
                    'transitions': transitions_after_win
                }
        
        if len(win_transitions) > 0:
            win_analysis[asset] = win_transitions
            
            print(f"  {asset:>6}: Analyzed {sum(wt['count'] for wt in win_transitions.values())} win transitions")
    
    print(f"✅ Analyzed win transitions for {len(win_analysis)} assets")
    
    return win_analysis

def calculate_momentum_metrics(asset_sequences):
    """Calculate advanced momentum and mean reversion metrics"""
    
    print(f"\n📊 Calculating advanced momentum metrics...")
    
    momentum_analysis = {}
    
    for asset, sequence_data in asset_sequences.items():
        states = sequence_data['states']
        returns = sequence_data['returns']
        days_held = sequence_data['days_held']
        
        if len(states) < 3:
            continue
        
        # Calculate various momentum metrics
        
        # 1. Return momentum (correlation between consecutive returns)
        if len(returns) >= 3:
            return_correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(set(returns)) > 1 else 0
        else:
            return_correlation = 0
        
        # 2. Speed momentum (do quick wins lead to more quick wins?)
        quick_trades = [i for i, days in enumerate(days_held) if days <= 90]
        quick_momentum = 0
        if len(quick_trades) >= 2:
            quick_pairs = sum(1 for i in range(len(quick_trades)-1) if quick_trades[i+1] == quick_trades[i] + 1)
            quick_momentum = quick_pairs / (len(quick_trades) - 1) if len(quick_trades) > 1 else 0
        
        # 3. State persistence (how often does the same state repeat?)
        state_repeats = sum(1 for i in range(len(states)-1) if states[i] == states[i+1])
        state_persistence = state_repeats / (len(states) - 1) if len(states) > 1 else 0
        
        # 4. Profit streak analysis
        profit_streaks = []
        loss_streaks = []
        current_profit_streak = 0
        current_loss_streak = 0
        
        for return_val in returns:
            if return_val > 0:
                current_profit_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_profit_streak > 0:
                    profit_streaks.append(current_profit_streak)
                    current_profit_streak = 0
        
        # Add final streak
        if current_profit_streak > 0:
            profit_streaks.append(current_profit_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        momentum_analysis[asset] = {
            'return_correlation': return_correlation,
            'quick_momentum': quick_momentum,
            'state_persistence': state_persistence,
            'avg_profit_streak': np.mean(profit_streaks) if profit_streaks else 0,
            'max_profit_streak': max(profit_streaks) if profit_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'total_profit_streaks': len(profit_streaks),
            'total_loss_streaks': len(loss_streaks),
            'overall_momentum_score': (return_correlation + quick_momentum + state_persistence) / 3
        }
        
        print(f"  {asset:>6}: Momentum score: {momentum_analysis[asset]['overall_momentum_score']:.3f}")
    
    print(f"✅ Calculated momentum metrics for {len(momentum_analysis)} assets")
    
    return momentum_analysis

# ============================================================================
# OUTPUT AND SAVING FUNCTIONS
# ============================================================================

def save_advanced_sequences(asset_sequences, folders):
    """Save advanced outcome sequences"""
    
    sequences_data = []
    
    for asset, sequence_data in asset_sequences.items():
        for i, (state, return_pct, days, date, period, max_ret, min_ret) in enumerate(zip(
            sequence_data['states'], 
            sequence_data['returns'],
            sequence_data['days_held'],
            sequence_data['dates'], 
            sequence_data['periods'],
            sequence_data['max_returns'],
            sequence_data['min_returns'])):
            
            sequences_data.append({
                'asset': asset,
                'sequence_order': i + 1,
                'period': period,
                'start_date': date,
                'performance_state': state,
                'return_pct': return_pct,
                'days_held': days,
                'max_return_pct': max_ret,
                'min_return_pct': min_ret,
                'drawdown_during_trade': max_ret - min_ret if pd.notna(max_ret) and pd.notna(min_ret) else 0,
                'total_trades': sequence_data['total_trades']
            })
    
    sequences_df = pd.DataFrame(sequences_data)
    sequences_path = os.path.join(folders['sequences'], 'advanced_outcome_sequences.csv')
    sequences_df.to_csv(sequences_path, index=False)
    
    print(f"💾 Saved advanced sequences: {sequences_path} ({len(sequences_data)} records)")
    
    return sequences_path

def save_drawdown_recovery_analysis(recovery_analysis, folders):
    """Save drawdown recovery analysis"""
    
    recovery_data = []
    detailed_recovery = []
    
    for asset, analysis in recovery_analysis.items():
        # Summary data
        recovery_data.append({
            'asset': asset,
            'total_drawdowns': analysis['total_drawdowns'],
            'recovery_rate': analysis['recovery_rate'],
            'quick_recovery_rate': analysis['quick_recovery_rate'],
            'strong_recovery_rate': analysis['strong_recovery_rate'],
            'avg_recovery_return': analysis['avg_recovery_return'],
            'avg_recovery_days': analysis['avg_recovery_days']
        })
        
        # Detailed recovery patterns
        for i, pattern in enumerate(analysis['recovery_patterns']):
            detailed_recovery.append({
                'asset': asset,
                'drawdown_sequence': i + 1,
                'drawdown_return': pattern['drawdown_return'],
                'next_state': pattern['next_state'],
                'next_return': pattern['next_return'],
                'next_days': pattern['next_days'],
                'is_quick_recovery': pattern['is_quick_recovery'],
                'is_strong_recovery': pattern['is_strong_recovery'],
                'is_any_recovery': pattern['is_any_recovery']
            })
    
    # Save summary
    recovery_df = pd.DataFrame(recovery_data)
    recovery_path = os.path.join(folders['drawdown'], 'drawdown_recovery_summary.csv')
    recovery_df.to_csv(recovery_path, index=False)
    
    # Save detailed patterns
    detailed_df = pd.DataFrame(detailed_recovery)
    detailed_path = os.path.join(folders['drawdown'], 'detailed_recovery_patterns.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    print(f"💾 Saved drawdown analysis: {recovery_path} and {detailed_path}")
    
    return recovery_path, detailed_path

def save_win_quality_analysis(win_analysis, folders):
    """Save win quality transition analysis"""
    
    win_data = []
    
    for asset, win_transitions in win_analysis.items():
        for win_type, metrics in win_transitions.items():
            win_data.append({
                'asset': asset,
                'win_type': win_type,
                'count': metrics['count'],
                'next_profit_rate': metrics['next_profit_rate'],
                'next_quick_rate': metrics['next_quick_rate'],
                'next_big_win_rate': metrics['next_big_win_rate'],
                'avg_next_return': metrics['avg_next_return'],
                'avg_next_days': metrics['avg_next_days']
            })
    
    win_df = pd.DataFrame(win_data)
    win_path = os.path.join(folders['quartiles'], 'win_quality_transitions.csv')
    win_df.to_csv(win_path, index=False)
    
    print(f"💾 Saved win quality analysis: {win_path}")
    
    return win_path

def save_momentum_analysis(momentum_analysis, folders):
    """Save momentum and mean reversion analysis"""
    
    momentum_data = []
    
    for asset, metrics in momentum_analysis.items():
        momentum_data.append({
            'asset': asset,
            'return_correlation': metrics['return_correlation'],
            'quick_momentum': metrics['quick_momentum'],
            'state_persistence': metrics['state_persistence'],
            'avg_profit_streak': metrics['avg_profit_streak'],
            'max_profit_streak': metrics['max_profit_streak'],
            'avg_loss_streak': metrics['avg_loss_streak'],
            'max_loss_streak': metrics['max_loss_streak'],
            'total_profit_streaks': metrics['total_profit_streaks'],
            'total_loss_streaks': metrics['total_loss_streaks'],
            'overall_momentum_score': metrics['overall_momentum_score']
        })
    
    momentum_df = pd.DataFrame(momentum_data)
    momentum_path = os.path.join(folders['analysis'], 'momentum_analysis.csv')
    momentum_df.to_csv(momentum_path, index=False)
    
    print(f"💾 Saved momentum analysis: {momentum_path}")
    
    return momentum_path

def create_comprehensive_report(recovery_analysis, win_analysis, momentum_analysis, state_info, folders):
    """Create comprehensive analysis report"""
    
    report_path = os.path.join(folders['analysis'], 'advanced_markov_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("ADVANCED STRADDLE MARKOV ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # State distribution
        f.write("PERFORMANCE STATE DISTRIBUTION:\n")
        f.write("-" * 35 + "\n")
        for state, count in state_info['state_counts'].items():
            pct = (count / state_info['state_counts'].sum()) * 100
            f.write(f"{state:20}: {count:4d} ({pct:5.1f}%)\n")
        
        f.write(f"\nProfit Quartile Thresholds:\n")
        f.write(f"Q1 (Small Win): 0% to {state_info['profit_quartiles'][0]:.1f}%\n")
        f.write(f"Q2 (Medium Win): {state_info['profit_quartiles'][0]:.1f}% to {state_info['profit_quartiles'][1]:.1f}%\n")
        f.write(f"Q3 (Good Win): {state_info['profit_quartiles'][1]:.1f}% to {state_info['profit_quartiles'][2]:.1f}%\n")
        f.write(f"Q4 (Big Win): {state_info['profit_quartiles'][2]:.1f}%+\n")
        f.write(f"Big Drawdown Threshold: {state_info['loss_threshold']:.1f}% or worse\n")
        
        # Drawdown recovery analysis
        if recovery_analysis:
            f.write(f"\nDRAWDOWN RECOVERY ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            recovery_df = pd.DataFrame([
                {
                    'asset': asset,
                    'total_drawdowns': analysis['total_drawdowns'],
                    'recovery_rate': analysis['recovery_rate'],
                    'quick_recovery_rate': analysis['quick_recovery_rate'],
                    'strong_recovery_rate': analysis['strong_recovery_rate'],
                    'avg_recovery_return': analysis['avg_recovery_return'],
                    'avg_recovery_days': analysis['avg_recovery_days']
                }
                for asset, analysis in recovery_analysis.items()
            ])
            
            # Overall statistics
            f.write(f"Total assets with drawdowns: {len(recovery_analysis)}\n")
            f.write(f"Average recovery rate: {recovery_df['recovery_rate'].mean()*100:.1f}%\n")
            f.write(f"Average quick recovery rate: {recovery_df['quick_recovery_rate'].mean()*100:.1f}%\n")
            f.write(f"Average strong recovery rate: {recovery_df['strong_recovery_rate'].mean()*100:.1f}%\n")
            f.write(f"Average recovery return: {recovery_df['avg_recovery_return'].mean():+.1f}%\n")
            f.write(f"Average recovery time: {recovery_df['avg_recovery_days'].mean():.0f} days\n\n")
            
            # Top recovery assets
            top_recovery = recovery_df.nlargest(10, 'recovery_rate')
            f.write("Top 10 Assets by Recovery Rate:\n")
            for i, (_, row) in enumerate(top_recovery.iterrows()):
                f.write(f"{i+1:2d}. {row['asset']:>6}: {row['recovery_rate']*100:5.1f}% recovery, "
                       f"{row['avg_recovery_return']:+5.1f}% avg return, "
                       f"{row['avg_recovery_days']:3.0f} days\n")
            
            # Best quick recovery assets
            top_quick_recovery = recovery_df.nlargest(10, 'quick_recovery_rate')
            f.write(f"\nTop 10 Assets by Quick Recovery Rate:\n")
            for i, (_, row) in enumerate(top_quick_recovery.iterrows()):
                f.write(f"{i+1:2d}. {row['asset']:>6}: {row['quick_recovery_rate']*100:5.1f}% quick recovery, "
                       f"{row['total_drawdowns']:2.0f} drawdowns\n")
        
        # Win quality analysis
        if win_analysis:
            f.write(f"\nWIN QUALITY TRANSITION ANALYSIS:\n")
            f.write("-" * 35 + "\n")
            
            # Aggregate win quality data across all assets
            win_summary = {}
            win_types = ['SMALL_WINS', 'MEDIUM_WINS', 'GOOD_WINS', 'BIG_WINS']
            
            for win_type in win_types:
                profit_rates = []
                quick_rates = []
                big_win_rates = []
                avg_returns = []
                avg_days = []
                
                for asset, win_transitions in win_analysis.items():
                    if win_type in win_transitions:
                        wt = win_transitions[win_type]
                        profit_rates.append(wt['next_profit_rate'])
                        quick_rates.append(wt['next_quick_rate'])
                        big_win_rates.append(wt['next_big_win_rate'])
                        avg_returns.append(wt['avg_next_return'])
                        avg_days.append(wt['avg_next_days'])
                
                if profit_rates:
                    win_summary[win_type] = {
                        'assets_count': len(profit_rates),
                        'avg_next_profit_rate': np.mean(profit_rates),
                        'avg_next_quick_rate': np.mean(quick_rates),
                        'avg_next_big_win_rate': np.mean(big_win_rates),
                        'avg_next_return': np.mean(avg_returns),
                        'avg_next_days': np.mean(avg_days)
                    }
            
            # Display win quality patterns
            f.write("After different win types, what happens next:\n\n")
            for win_type, summary in win_summary.items():
                win_label = win_type.replace('_', ' ').title()
                f.write(f"{win_label}:\n")
                f.write(f"  Assets with this pattern: {summary['assets_count']}\n")
                f.write(f"  Next profit rate: {summary['avg_next_profit_rate']*100:.1f}%\n")
                f.write(f"  Next quick win rate: {summary['avg_next_quick_rate']*100:.1f}%\n")
                f.write(f"  Next big win rate: {summary['avg_next_big_win_rate']*100:.1f}%\n")
                f.write(f"  Avg next return: {summary['avg_next_return']:+.1f}%\n")
                f.write(f"  Avg next days: {summary['avg_next_days']:.0f}\n\n")
            
            # Key insights
            f.write("KEY WIN QUALITY INSIGHTS:\n")
            if 'SMALL_WINS' in win_summary and 'BIG_WINS' in win_summary:
                small_profit = win_summary['SMALL_WINS']['avg_next_profit_rate']
                big_profit = win_summary['BIG_WINS']['avg_next_profit_rate']
                small_return = win_summary['SMALL_WINS']['avg_next_return']
                big_return = win_summary['BIG_WINS']['avg_next_return']
                
                if big_profit > small_profit + 0.05:
                    f.write("→ BIG WINS tend to be followed by MORE profits (Success breeds success)\n")
                elif small_profit > big_profit + 0.05:
                    f.write("→ SMALL WINS tend to be followed by MORE profits (Consistency wins)\n")
                else:
                    f.write("→ Win size doesn't strongly predict next profit probability\n")
                
                if big_return > small_return + 2.0:
                    f.write("→ BIG WINS tend to be followed by BIGGER next returns\n")
                elif small_return > big_return + 2.0:
                    f.write("→ SMALL WINS tend to be followed by BIGGER next returns (mean reversion)\n")
                else:
                    f.write("→ Win size doesn't strongly predict next return magnitude\n")
        
        # Momentum analysis
        if momentum_analysis:
            f.write(f"\nMOMENTUM ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            momentum_df = pd.DataFrame([
                {
                    'asset': asset,
                    'overall_momentum_score': metrics['overall_momentum_score'],
                    'return_correlation': metrics['return_correlation'],
                    'quick_momentum': metrics['quick_momentum'],
                    'state_persistence': metrics['state_persistence'],
                    'max_profit_streak': metrics['max_profit_streak'],
                    'max_loss_streak': metrics['max_loss_streak'],
                    'avg_profit_streak': metrics['avg_profit_streak']
                }
                for asset, metrics in momentum_analysis.items()
            ])
            
            # Overall momentum statistics
            f.write(f"Assets analyzed: {len(momentum_analysis)}\n")
            f.write(f"Average momentum score: {momentum_df['overall_momentum_score'].mean():.3f}\n")
            f.write(f"Average return correlation: {momentum_df['return_correlation'].mean():+.3f}\n")
            f.write(f"Assets with positive momentum (>0.2): {(momentum_df['overall_momentum_score'] > 0.2).sum()}\n")
            f.write(f"Assets with mean reversion (<-0.1): {(momentum_df['overall_momentum_score'] < -0.1).sum()}\n")
            f.write(f"Average max profit streak: {momentum_df['max_profit_streak'].mean():.1f}\n")
            f.write(f"Average max loss streak: {momentum_df['max_loss_streak'].mean():.1f}\n\n")
            
            # Top momentum assets
            top_momentum = momentum_df.nlargest(10, 'overall_momentum_score')
            f.write("Top 10 Assets by Momentum Score:\n")
            for i, (_, row) in enumerate(top_momentum.iterrows()):
                f.write(f"{i+1:2d}. {row['asset']:>6}: {row['overall_momentum_score']:5.3f} momentum, "
                       f"{row['return_correlation']:+5.3f} correlation, "
                       f"{row['max_profit_streak']:.0f} max streak\n")
            
            # Most mean reverting assets
            top_mean_reversion = momentum_df.nsmallest(10, 'overall_momentum_score')
            f.write(f"\nTop 10 Mean Reverting Assets:\n")
            for i, (_, row) in enumerate(top_mean_reversion.iterrows()):
                f.write(f"{i+1:2d}. {row['asset']:>6}: {row['overall_momentum_score']:5.3f} momentum, "
                       f"{row['return_correlation']:+5.3f} correlation\n")
        
        # Summary insights
        f.write(f"\nSUMMARY INSIGHTS:\n")
        f.write("-" * 20 + "\n")
        
        if recovery_analysis:
            avg_recovery = np.mean([analysis['recovery_rate'] for analysis in recovery_analysis.values()])
            avg_quick_recovery = np.mean([analysis['quick_recovery_rate'] for analysis in recovery_analysis.values()])
            
            if avg_recovery > 0.7:
                f.write("1. DRAWDOWN RECOVERY: Strong - Most drawdowns are followed by profitable trades\n")
            elif avg_recovery > 0.5:
                f.write("1. DRAWDOWN RECOVERY: Moderate - Mixed recovery patterns after drawdowns\n")
            else:
                f.write("1. DRAWDOWN RECOVERY: Weak - Drawdowns often followed by more poor performance\n")
            
            if avg_quick_recovery > 0.4:
                f.write("2. RECOVERY SPEED: Fast - Drawdowns often followed by quick recoveries\n")
            else:
                f.write("2. RECOVERY SPEED: Slow - Recoveries take time to materialize\n")
        
        if momentum_analysis:
            avg_momentum = np.mean([metrics['overall_momentum_score'] for metrics in momentum_analysis.values()])
            
            if avg_momentum > 0.2:
                f.write("3. OVERALL PATTERN: Strong MOMENTUM - Performance tends to cluster\n")
            elif avg_momentum < -0.1:
                f.write("3. OVERALL PATTERN: Strong MEAN REVERSION - Performance tends to alternate\n")
            else:
                f.write("3. OVERALL PATTERN: Mixed - No dominant momentum or reversion pattern\n")
        
        f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📋 Saved comprehensive report: {report_path}")
    
    return report_path

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to run advanced Markov analysis"""
    
    print("🎯 ADVANCED STRADDLE MARKOV ANALYSIS")
    print("="*80)
    print("Sophisticated analysis using quartile-based states and drawdown recovery")
    print("Key Questions:")
    print("  1. After big drawdowns, are next trades quicker/better?")
    print("  2. Do big wins lead to better subsequent performance?")
    print("  3. What are the momentum vs mean reversion patterns?")
    print("="*80)
    
    # Configuration - find most recent early exit results
    summary_file = input("Enter path to early_exit_straddle_summary.csv (or press Enter for auto-search): ").strip()
    
    if not summary_file:
        # Try to find the most recent early exit summary file
        import glob
        search_pattern = "straddle_early_exit_*/summaries/early_exit_straddle_summary.csv"
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            # Get the most recent file
            summary_file = max(matching_files, key=os.path.getctime)
            print(f"📁 Auto-found: {summary_file}")
        else:
            print("❌ No early exit summary files found. Please run B_02_straddle_early_exit.py first.")
            return
    
    # Step 1: Create organized folder structure
    folders = create_output_folders()
    
    # Step 2: Load early exit results
    df = load_early_exit_results(summary_file)
    if df is None:
        return
    
    # Step 3: Define sophisticated performance states
    df_with_states, state_info = define_performance_states(df)
    if df_with_states is None:
        return
    
    # Step 4: Create advanced outcome sequences
    asset_sequences = create_advanced_outcome_sequences(df_with_states)
    if len(asset_sequences) == 0:
        print("❌ No valid asset sequences created")
        return
    
    # Step 5: Analyze drawdown recovery patterns
    recovery_analysis = analyze_drawdown_recovery_patterns(asset_sequences)
    
    # Step 6: Analyze win quality transitions
    win_analysis = analyze_win_quality_transitions(asset_sequences)
    
    # Step 7: Calculate momentum metrics
    momentum_analysis = calculate_momentum_metrics(asset_sequences)
    
    # Step 8: Save all results
    print(f"\n💾 Saving advanced Markov analysis results...")
    
    sequences_path = save_advanced_sequences(asset_sequences, folders)
    
    if recovery_analysis:
        recovery_paths = save_drawdown_recovery_analysis(recovery_analysis, folders)
    
    if win_analysis:
        win_path = save_win_quality_analysis(win_analysis, folders)
    
    if momentum_analysis:
        momentum_path = save_momentum_analysis(momentum_analysis, folders)
    
    # Create comprehensive report
    report_path = create_comprehensive_report(recovery_analysis, win_analysis, momentum_analysis, state_info, folders)
    
    # Step 9: Display key findings
    print(f"\n" + "="*80)
    print(f"🎉 ADVANCED MARKOV ANALYSIS COMPLETE!")
    print("="*80)
    
    # Key findings summary
    print(f"\n🎯 KEY FINDINGS:")
    
    # Drawdown recovery insights
    if recovery_analysis:
        total_assets_with_drawdowns = len(recovery_analysis)
        avg_recovery_rate = np.mean([analysis['recovery_rate'] for analysis in recovery_analysis.values()])
        avg_quick_recovery = np.mean([analysis['quick_recovery_rate'] for analysis in recovery_analysis.values()])
        avg_strong_recovery = np.mean([analysis['strong_recovery_rate'] for analysis in recovery_analysis.values()])
        
        print(f"\n📉 DRAWDOWN RECOVERY:")
        print(f"   Assets with drawdowns: {total_assets_with_drawdowns}")
        print(f"   Average recovery rate: {avg_recovery_rate*100:.1f}%")
        print(f"   Average quick recovery rate: {avg_quick_recovery*100:.1f}%")
        print(f"   Average strong recovery rate: {avg_strong_recovery*100:.1f}%")
        
        if avg_quick_recovery > 0.5:
            print(f"   🚀 INSIGHT: Drawdowns tend to be followed by QUICK recoveries")
        elif avg_recovery_rate > 0.7:
            print(f"   📈 INSIGHT: Drawdowns tend to be followed by eventual recoveries")
        else:
            print(f"   ⚠️  INSIGHT: Drawdowns show mixed recovery patterns")
    
    # Win quality insights
    if win_analysis:
        # Aggregate win transition data
        all_small_next_profit = []
        all_big_next_profit = []
        all_small_next_return = []
        all_big_next_return = []
        
        for asset_wins in win_analysis.values():
            if 'SMALL_WINS' in asset_wins:
                all_small_next_profit.append(asset_wins['SMALL_WINS']['next_profit_rate'])
                all_small_next_return.append(asset_wins['SMALL_WINS']['avg_next_return'])
            if 'BIG_WINS' in asset_wins:
                all_big_next_profit.append(asset_wins['BIG_WINS']['next_profit_rate'])
                all_big_next_return.append(asset_wins['BIG_WINS']['avg_next_return'])
        
        if all_small_next_profit and all_big_next_profit:
            small_profit_avg = np.mean(all_small_next_profit)
            big_profit_avg = np.mean(all_big_next_profit)
            small_return_avg = np.mean(all_small_next_return)
            big_return_avg = np.mean(all_big_next_return)
            
            print(f"\n🏆 WIN QUALITY PATTERNS:")
            print(f"   After small wins: {small_profit_avg*100:.1f}% next profit rate, {small_return_avg:+.1f}% avg return")
            print(f"   After big wins: {big_profit_avg*100:.1f}% next profit rate, {big_return_avg:+.1f}% avg return")
            
            if big_profit_avg > small_profit_avg + 0.05:
                print(f"   🔥 INSIGHT: Big wins breed MORE success (momentum in profits)")
            elif small_profit_avg > big_profit_avg + 0.05:
                print(f"   ⚖️  INSIGHT: Small wins breed MORE success (consistency wins)")
            else:
                print(f"   📊 INSIGHT: Win size doesn't strongly predict next profit probability")
                
            if big_return_avg > small_return_avg + 2.0:
                print(f"   💰 INSIGHT: Big wins lead to BIGGER next returns")
            elif small_return_avg > big_return_avg + 2.0:
                print(f"   🔄 INSIGHT: Small wins lead to BIGGER next returns (reversion)")
            else:
                print(f"   📊 INSIGHT: Win size doesn't strongly predict next return size")
    
    # Momentum insights
    if momentum_analysis:
        momentum_scores = [metrics['overall_momentum_score'] for metrics in momentum_analysis.values()]
        avg_momentum = np.mean(momentum_scores)
        high_momentum_assets = sum(1 for score in momentum_scores if score > 0.2)
        mean_reversion_assets = sum(1 for score in momentum_scores if score < -0.1)
        
        print(f"\n📊 MOMENTUM PATTERNS:")
        print(f"   Average momentum score: {avg_momentum:.3f}")
        print(f"   High momentum assets: {high_momentum_assets}/{len(momentum_analysis)}")
        print(f"   Mean reversion assets: {mean_reversion_assets}/{len(momentum_analysis)}")
        
        if avg_momentum > 0.2:
            print(f"   🎯 INSIGHT: Strong MOMENTUM patterns detected")
        elif avg_momentum < -0.1:
            print(f"   ↩️  INSIGHT: Strong MEAN REVERSION patterns detected")
        else:
            print(f"   🎲 INSIGHT: Mixed momentum/reversion patterns")
    
    # Top performing assets for different criteria
    if recovery_analysis and len(recovery_analysis) >= 5:
        best_recovery_assets = sorted(recovery_analysis.items(), 
                                    key=lambda x: x[1]['recovery_rate'], reverse=True)[:5]
        print(f"\n🥇 TOP 5 DRAWDOWN RECOVERY ASSETS:")
        for i, (asset, analysis) in enumerate(best_recovery_assets):
            print(f"   {i+1}. {asset}: {analysis['recovery_rate']*100:.0f}% recovery rate, {analysis['avg_recovery_return']:+.1f}% avg return")
    
    if momentum_analysis and len(momentum_analysis) >= 5:
        best_momentum_assets = sorted(momentum_analysis.items(), 
                                    key=lambda x: x[1]['overall_momentum_score'], reverse=True)[:5]
        print(f"\n🚀 TOP 5 MOMENTUM ASSETS:")
        for i, (asset, metrics) in enumerate(best_momentum_assets):
            print(f"   {i+1}. {asset}: {metrics['overall_momentum_score']:.3f} momentum score, {metrics['max_profit_streak']:.0f} max streak")
    
    print(f"\n📁 OUTPUT FILES GENERATED:")
    print(f"   📈 {sequences_path}")
    if recovery_analysis:
        print(f"   📉 Drawdown recovery analysis in {folders['drawdown']}/")
    if win_analysis:
        print(f"   🏆 Win quality analysis in {folders['quartiles']}/")
    if momentum_analysis:
        print(f"   📊 Momentum analysis in {folders['analysis']}/")
    print(f"   📋 {report_path}")
    
    print(f"\n✅ Advanced Markov analysis complete!")
    print(f"💡 Check the comprehensive report for detailed insights!")
    print(f"🎯 Key questions answered about drawdown recovery and win patterns!")
    print("="*80)

if __name__ == "__main__":
    main()