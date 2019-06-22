# Non-local means Filter
### 1. Bilateral Filter
Bilateral filter is an algorithm that consider spatial information and pixel value informaiton in the same time. That is why it is so-called "Bi"-lateral. 

* In natural image, the variation in spatial domain is low which means there are high correlation between the central pixel and its neighborhoods. 

* However, this assumption will greatly blur the results in the edge boundary which lost the edge features. Therefore, we compensate the flawness via the value of the pixels. In the boundary, bilateral give greatly  different weights to two regions which makes the central pixel based  more on the pixels that belong to its region.

Below are the formula of Bilateral filter:

![](https://i.imgur.com/6l6HG8E.png)

where 

![](https://i.imgur.com/yTb1GgQ.png)


![](https://i.imgur.com/7ceoVlS.png)

The above formula is derived from ![](https://i.imgur.com/0cYmFOP.png)
which is clear that the weight is based on space factor and pixel value factor.

---

### 2. NL-means filter
NL-means filter is an extension of bilateral filter, where instead of averaging values of pixels with similar values, **the values of pixels centered on similar patches are averaged**.

The formula for NL-means can be derived from ![](https://i.imgur.com/oUKcXpI.png)

There is only a slightly difference between bilateral filter and NL-means filter. The formula can be written as below:
![](https://i.imgur.com/bDLsdaF.png)

![](https://i.imgur.com/bjnK2lc.png)

where 

![](https://i.imgur.com/yTb1GgQ.png)

![](https://i.imgur.com/qeYrPJW.png)

In this program, it improves the computation time via **image shift**.
(similar to how we improve convolution)




