# LSAMP Research Fall 2019

## Feature Points

### Investigated

Curb Stomping
- Stepping down from (or up onto) a curb. With phone in pocket, this produces changes in Y component as expected, but you also get some acceleration in Z as you're moving forward when stepping down from/stepping up onto a curb.

Stationary vs. Non-Stationary

* There is an expected pattern for normal walking. When you stop walking, acceleration goes to zero.

### Unknown

* Is there a general curve that can be extracted from accel data when walking normally?
  * What does acceleration near crossings do to this curve?
* Turn at curb or jaywalking model?
  * Walking curve along as normal, maybe slow down a bit, then turn to face street from sidewalk/curb.
* Gyroscope on ramps.
* Magnetometer near cars, does this produce a doppler like effect, but in terms of a compass and a car obstructing a magnetic field? Would this be the case with electric motors?

### Plan of Action

Build out several of the feature points in app then combine into NN.