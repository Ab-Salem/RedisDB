# pip install redis numpy
import redis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import time

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Function to calculate cosine similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Clear previous data
r.flushall()

# Charm attributes:
# - power_level: magical strength of the charm (0-1)
# - protection_level: defensive capabilities (0-1)
# - complexity: how intricate the charm is (0-1)
# - duration: how long the effects last (0-1)

# Charm data
charms = {
    "Shield": np.array([0.4, 0.9, 0.5, 0.7], dtype=np.float32),  # Strong protection, moderate power
    "Fireball": np.array([0.8, 0.2, 0.6, 0.4], dtype=np.float32),  # High power, low protection
    "Healing": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),  # Balanced with long duration
    "Invisibility": np.array([0.3, 0.7, 0.9, 0.5], dtype=np.float32),  # Complex, good protection
    "Lightning": np.array([0.9, 0.1, 0.5, 0.2], dtype=np.float32),  # Very powerful, short duration
    "Levitation": np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32),  # Balanced charm
    "Ice": np.array([0.7, 0.6, 0.4, 0.5], dtype=np.float32),  # Good power and protection
    "Teleport": np.array([0.6, 0.3, 0.8, 0.1], dtype=np.float32),  # Complex with short duration
}

# Add charms to Redis
for charm_name, vector in charms.items():
    r.set(f"charm:{charm_name}", vector.tobytes())
    print(f"Added charm: {charm_name} with vector: {vector}")

def find_similar_charms(query_vector, top_n=3):
    """Find the most similar charms to the query vector."""
    results = []
    for charm_name in charms.keys():
        charm_vector_bytes = r.get(f"charm:{charm_name}")
        charm_vector = np.frombuffer(charm_vector_bytes, dtype=np.float32)
        similarity = cosine_similarity(query_vector, charm_vector)
        results.append((charm_name, similarity))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N results
    return results[:top_n]

def add_new_charm(name, power, protection, complexity, duration):
    """Add a new charm to the database."""
    vector = np.array([power, protection, complexity, duration], dtype=np.float32)
    r.set(f"charm:{name}", vector.tobytes())
    charms[name] = vector
    return name, vector

def visualize_charms_2d(highlight_charm=None, query_vector=None):
    """
    Create a 2D visualization of charms based on power and protection levels.
    Optionally highlight a specific charm and show a query vector.
    """
    plt.figure(figsize=(10, 8))
    
    # Extract power (x) and protection (y) from all charms
    x = [vector[0] for vector in charms.values()]
    y = [vector[1] for vector in charms.values()]
    names = list(charms.keys())
    
    # Plot all charms
    plt.scatter(x, y, c='blue', alpha=0.7, s=100)
    
    # Label each point with charm name
    for i, name in enumerate(names):
        plt.annotate(name, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Highlight a specific charm if requested
    if highlight_charm and highlight_charm in charms:
        highlight_idx = names.index(highlight_charm)
        plt.scatter(x[highlight_idx], y[highlight_idx], c='green', s=200, edgecolor='black')
    
    # Show query vector if provided
    if query_vector is not None:
        plt.scatter(query_vector[0], query_vector[1], c='red', marker='x', s=200, label='Query')
        
        # Draw arrows from query to top 3 matches
        matches = find_similar_charms(query_vector, top_n=3)
        for charm_name, similarity in matches:
            charm_idx = names.index(charm_name)
            arrow = FancyArrowPatch((query_vector[0], query_vector[1]), 
                                   (x[charm_idx], y[charm_idx]),
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   color='gray', 
                                   alpha=0.6)
            plt.gca().add_patch(arrow)
            # Add similarity score
            mid_x = (query_vector[0] + x[charm_idx]) / 2
            mid_y = (query_vector[1] + y[charm_idx]) / 2
            plt.annotate(f"{similarity:.2f}", (mid_x, mid_y), 
                         xytext=(5, 5), textcoords='offset points', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Power Level')
    plt.ylabel('Protection Level')
    plt.title('Charm Vector Space (Power vs Protection)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                 label='Similarity score')
    plt.tight_layout()
    plt.savefig('charm_visualization.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example 1: Query for a balanced charm with good protection
    query_vector = np.array([0.5, 0.7, 0.6, 0.6], dtype=np.float32)
    print(f"\nQuery Vector (Balanced with good protection): {query_vector}")
    
    results = find_similar_charms(query_vector)
    
    print("\nSimilarity Search Results:")
    for charm_name, similarity in results:
        print(f"Charm: {charm_name}, Similarity: {similarity:.4f}")
    
    # Visualize the results
    visualize_charms_2d(results[0][0], query_vector)
    
    # Example 2: Add a new charm and query for high power spells
    print("\nAdding new charm: Earthquake")
    new_charm = add_new_charm("Earthquake", 0.9, 0.3, 0.7, 0.4)
    print(f"Added: {new_charm}")
    
    time.sleep(1)  # Just for demonstration
    
    query_vector2 = np.array([0.8, 0.2, 0.5, 0.3], dtype=np.float32)
    print(f"\nQuery Vector (High power, low protection): {query_vector2}")
    
    results2 = find_similar_charms(query_vector2)
    
    print("\nSimilarity Search Results:")
    for charm_name, similarity in results2:
        print(f"Charm: {charm_name}, Similarity: {similarity:.4f}")
    
    # Visualize the new results
    visualize_charms_2d(results2[0][0], query_vector2)
