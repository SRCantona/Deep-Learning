
# Gold Price Prediction Using Parallel Programming

## Project Overview
This project aims to build a machine learning model that predicts gold prices by leveraging parallel programming techniques. The project demonstrates the benefits and challenges of parallel execution, focusing on race conditions, synchronization, and performance improvements.

---

## Key Concepts

### 1. Race Condition Analysis
In the initial implementation, race conditions were identified when modifying shared variables concurrently. This led to unpredictable and inconsistent results.

**Example:**
```python
gold_price = 1500

def increment_price():
    global gold_price
    for _ in range(100000):
        gold_price += 1

def decrement_price():
    global gold_price
    for _ in range(100000):
        gold_price -= 1
```
**Explanation:**  
- Multiple threads simultaneously modify `gold_price` without synchronization.  
- This creates a race condition where increments and decrements overlap, resulting in incorrect final values.  

---

### 2. Synchronization and Performance Testing
To address the race condition, a lock mechanism (`threading.Lock`) was implemented. This ensures only one thread accesses the shared resource at a time.

**Example (Using Lock):**
```python
from threading import Lock
lock = Lock()

def increment_price_safe():
    global gold_price
    with lock:
        for _ in range(100000):
            gold_price += 1
```
**Outcome:**  
- The lock prevents overlapping operations, ensuring consistent results.  
- However, the execution time increases slightly due to thread serialization.

---

### 3. Performance Comparison
The project evaluated performance with and without synchronization across different core counts (1, 2, 4, 8 cores).  
- **Without Lock:** Fast execution but data corruption.  
- **With Lock:** Consistent results but slower execution.  
- **Future Approach:** Atomic operations or reduction techniques may provide faster and safer alternatives.  

---

## Results Summary
- **Execution Time (Without Lock):** ~0.0886 seconds (inconsistent results).  
- **Execution Time (With Lock):** ~0.1500 seconds (consistent results).  
- **Recommendation:** Use locks for small-scale projects and explore reduction/atomic operations for larger core counts.  

---

## Notes
This README highlights critical sections of the code and their impact on performance. The full implementation, including model training and evaluation, can be found in the accompanying notebook.
