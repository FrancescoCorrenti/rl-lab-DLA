def compare_agents(*agents):
    """Compare multiple REINFORCE agents' training results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    if not agents:
        raise ValueError("At least one agent must be provided for comparison.")
    
    plt.figure(figsize=(14, 6))
    
    for agent in agents:
        if not hasattr(agent, 'evaluation_results') or not agent.evaluation_results:
            raise RuntimeError(f"Agent {agent} has no evaluation results. Run train_online() first.")
        
        df = pd.DataFrame(agent.evaluation_results)
        sns.lineplot(data=df, x='episode', y='avg_reward', label=str(agent), marker='o', linewidth=2.5, markersize=6)
    
    plt.title('Comparison of REINFORCE Agents', fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
