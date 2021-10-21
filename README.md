# so69552983

Solution for [this SO question](https://stackoverflow.com/q/69552983/51685) by [R2D2](https://stackoverflow.com/users/4410444/r2d2):

```
I have a RGB image which is a map including roads and stations.

I am trying to find the closest station to each other that there is a road between those. The road is showing in black and the stations are Red. I added the name so it would be more understandable.

O -> M
A -> B
B -> C
E -> F
G -> H
G -> C
M -> N
.
. 
.
The distance between the two stations is calculated base on the length of the road between those.

Some ideas about to do solve this: I was thinking removing the station from the roads and we have disconnected roads, then use the contour to calculate the length of the road in pixels. But in some cases (like F or E) the station doesn't cover the road completely and do not discount the road to three different part.

Please let me know what you think, how you solve it.
```