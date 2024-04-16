# Plotting README

# Learning Objectives

1. **What is a plot?**
   A plot is a graphical representation of data, which shows the relationship between two or more variables.

2. **What is a scatter plot? line graph? bar graph? histogram?**
   - A scatter plot is a type of plot that displays values for typically two variables for a set of data.
   - A line graph is a type of chart which displays information as a series of data points connected by straight line segments.
   - A bar graph is a chart that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
   - A histogram is an approximate representation of the distribution of numerical data.

3. **What is matplotlib?**
   Matplotlib is a plotting library for the Python programming language. It provides an object-oriented API for embedding plots into applications.

4. **How to plot data with matplotlib**
   Here is a basic example:
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3, 4])
   plt.ylabel('some numbers')
   plt.show()

5. **How to label a plot**
   You can use the xlabel and ylabel functions to set labels for the x and y axis respectively. For example:
   ```python
   plt.xlabel('X Axis')
   plt.ylabel('Y Axis')

6. **How to scale an Axis**
   Sure, here's how you might write those answers in a README format:

```markdown
# Plotting README

## Learning Objectives

1. **What is a plot?**
   A plot is a graphical representation of data, which shows the relationship between two or more variables.

2. **What is a scatter plot? line graph? bar graph? histogram?**
   - A scatter plot is a type of plot that displays values for typically two variables for a set of data.
   - A line graph is a type of chart which displays information as a series of data points connected by straight line segments.
   - A bar graph is a chart that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.
   - A histogram is an approximate representation of the distribution of numerical data.

3. **What is matplotlib?**
   Matplotlib is a plotting library for the Python programming language. It provides an object-oriented API for embedding plots into applications.

4. **How to plot data with matplotlib**
   Here is a basic example:
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3, 4])
   plt.ylabel('some numbers')
   plt.show()
   ```

5. **How to label a plot**
   You can use the `xlabel` and `ylabel` functions to set labels for the x and y axis respectively. For example:
   ```python
   plt.xlabel('X Axis')
   plt.ylabel('Y Axis')
   ```

6. **How to scale an axis**
   You can use the `xlim` and `ylim` functions to set the limits of the x and y axis respectively. For example:
   ```python
   plt.xlim(0, 10)
   plt.ylim(0, 20)
   ```

7. **How to plot multiple sets of data at the same time**
   You can call the `plot` function multiple times before calling `show`. For example:
   ```python
   plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
   plt.plot([1, 2, 3, 4], [2, 3, 5, 7])
   plt.show()
   ```
