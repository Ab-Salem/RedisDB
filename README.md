# Charm Vector Database

This project implements a vector similarity search system for magical charms using Redis. It demonstrates how to use Redis as a vector database to find similar magical charms based on their attributes.

## Project Overview

The system represents each charm as a 4-dimensional vector with the following attributes:
1. **Power Level** (0-1): The magical strength of the charm
2. **Protection Level** (0-1): The defensive capabilities 
3. **Complexity** (0-1): How intricate the charm is
4. **Duration** (0-1): How long the effects last

When a query vector is provided, the system calculates the cosine similarity between the query and all stored charms, then returns the most similar charms.

## Requirements

- Python 3.6+
- Redis server
- Python packages:
  - redis
  - numpy
  - matplotlib

## Installation

1. Install required packages:
```bash
pip install redis numpy matplotlib
```

2. Start Redis server:
```bash
redis-server
```

3. Run the script:
```bash
python charm_vector_db.py
```

## Features

- Store charm vectors in Redis
- Find similar charms using cosine similarity
- Add new charms dynamically
- Visualize charms in 2D space based on power and protection levels
- Highlight similarity connections between query and results

## Example Usage

The script includes two example queries:
1. A balanced charm with good protection (`[0.5, 0.7, 0.6, 0.6]`)
2. A high-power charm with low protection (`[0.8, 0.2, 0.5, 0.3]`)

It also demonstrates adding a new charm ("Earthquake") and then finding similar charms.

## Output

The script generates visualizations showing the charms in 2D space with similarity connections to the query vector.

## Project Extension Ideas

1. Implement k-nearest neighbors search algorithm for larger datasets
2. Add more dimensions to the vectors for more detailed charm attributes
3. Create a simple web interface for query visualization
4. Implement a recommendation system based on user preferences
5. Add auth & permission system to protect special charms

## License

MIT
