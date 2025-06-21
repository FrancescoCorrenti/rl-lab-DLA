def compare_agents(*agents, title="Comparison of Agents", threshold=None):
    """Compare multiple agents' training results."""
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
        ax = sns.lineplot(data=df, x='episode', y='avg_reward', label=str(agent), marker='o', linewidth=2.5, markersize=6)

        # Add a star at the end of the line
        if agent.has_solved:
            last_episode = df['episode'].iloc[-1]
            last_reward = df['avg_reward'].iloc[-1]
            plt.plot(last_episode, last_reward, '*', markersize=20, color=ax.get_lines()[-1].get_color(), markeredgecolor='black')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
