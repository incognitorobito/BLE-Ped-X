# LSAMP Research Fall 2019

## Feature Points

### Investigated

Looking Both Ways Before Crossing
As a kid you were likely taught (or quickly learned) this basic behavior. Everyone should slow down and look both ways before crossing the street. We can visualize this by plotting the accelerometer readings for a short walk.

Figure 2: Looking Both Ways Before Crossing Visualization

As we begin walking there is an initial acceleration from a stationary position. Then we get into our normal stride, but as we get close to the intersection (roughly the center of the image above), we slow down to look (and in some cases may even be stationary for a few seconds) then speed back up to get across the street safely. This pattern diverges slightly from your normal walking pattern and therefore combined with GPS data to confirm proximity to an intersection, and/or another of the behaviors below, could be a key indicator of crossing.

Curb Your Enthusiasm
axis_device

Figure 3: Android Device Axis (Source)

Jaywalking pedestrians are by definition not crossing at the crosswalk. They are therefore crossing at another point in the road, and likely stepping off the curb to do so. The quick bits of acceleration from this movement are detectable via an accelerometer. Stepping down from the curb produces a change in the Y-component, while you get a slight acceleration in the Z-component as one moves forward.

About Face
As a pedestrian approaches a crosswalk, they may need to turn to face the street/crosswalk. This feature would require data from the gyroscopic along with GPS and compass data to confirm a pedestrian’s heading.

### Unknown

* Is there a general curve that can be extracted from accel data when walking normally?
  * What does acceleration near crossings do to this curve?
* Turn at curb or jaywalking model?
  * Walking curve along as normal, maybe slow down a bit, then turn to face street from sidewalk/curb.
* Gyroscope on ramps.
* Magnetometer near cars, does this produce a doppler like effect, but in terms of a compass and a car obstructing a magnetic field? Would this be the case with electric motors?

### Plan of Action

Build out several of the feature points in app then combine into NN.
