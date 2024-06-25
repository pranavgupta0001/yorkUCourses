# PHY432: Final Project
# Team: The Skywalkers
# Members: Simon Tebeck, Pranav Gupta
# April 2024

# Objective 2: Orbits Earth

# import packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("ggplot")
import tqdm

# integrators
# import integrators



# =============================================================================
# 1. Initialising parameters
# =============================================================================


# gravitational constant
G_gravity = 4*np.pi**2 *(1/365.256)**2 # astronomical units but per days not years
yr = 365.256 # days

# astronomical unit
au =  1.495978e8 # km (!!)

def AU(x): return x/au
def AU_inv(x): return x*au


# masses
# SI
mass_sun = 1.9885E+30 # kg
mass_earth = 5.9724E+24 # kg
mass_moon = 7.346E+22 # kg
mass_endor = 7.52E+23 # kg
# mass_bb =


# Astronomical units
# DS_A: Death Star (light estimation)
# DS_B: Death Star (heavy estimation)
mass = {'Sun': 1.,
        'Earth': mass_earth/mass_sun,
        'Moon': mass_moon/mass_sun,
        'Endor': mass_endor/mass_sun,
        'DS_A': 1e18/mass_sun,
        'DS_B': 2.8e23/mass_sun
        }

# orbital period in Earth days
period = {'Earth': 365.256
          }

# speeds in AU/day
# for earth: use pre-determined speed at perihelion: 30.29 km/s
speed_earth = 30.29 * 3600*24 / au # speed earth around sun at perihelion
# for moon: speed at apogee: 0.966 km/s
speed_moon =  0.966 * 3600*24 / au # speed moon around earth at apogee
# for DS: number that lead to stable orbits
speed_DS = 3.99 * 3600*24 / au # speed DS around earth


# distance from the respective orbit center in AU
# note: Earth distance at perihelion (closest), Moon at apogee (farthest)
distance = {'Earth': 1.47098074e8 / au, # e8 is in km!!
            'Moon': 4.055e5 / au,
            'DS': 2.5e4 / au,
            }

# radii of the objects
radius = {'Sun':696000 / au,
          'Earth': 6357 / au,
          'Moon': 1737 / au,
          'Endor': 2450 / au,
          'BB': 74000 / au,
          'DS': 100 / au
          }


# =============================================================================
# 2. Functions for the orbit calculations (Solar System) (for Endor System
# see other file in Submission)
# =============================================================================

# note: Some of the following functions were inspired by HW07, code templates
# offered by Oliver Beckstein, modified by Simon Tebeck


def initial_position(distance, angle=0):
    """Calculate initial planet position.

    Parameters
    ----------
    angle : float
       initial angle relative to x axis (in degrees)
    distance : float
       initial distane from sun (in AU)

    Returns
    -------
    array
       position (x, y)
    """
    x = np.deg2rad(angle)
    return distance * np.array([np.cos(x), np.sin(x)])



def F_gravity(r, m, M):
    """Force due to gravity between two masses.

    Parameters
    ----------
    r : array
      distance vector (x, y)
    m, M : float
      masses of the two bodies

    Returns
    -------
    array
       force due to gravity (along r)
    """
    rr = np.sum(r * r)
    rhat = r / np.sqrt(rr)
    force_magnitude = - G_gravity * m * M / rr* rhat
    return force_magnitude


def omega(v, r):
    """Calculate angular velocity.

    The angular velocity is calculated as
    .. math::

          \omega = \frac{|\vec{v}|}{|\vec{r}|}

    Parameters
    ----------
    v : array
       velocity vectors for all N time steps; this
       should be a (N, dim) array
    r : array
       position vectors (N, dim) array

    Returns
    -------
    array
       angular velocity for each time step as 1D array of
       length N
    """
    speed = np.linalg.norm(v, axis=1)
    distance = np.linalg.norm(r, axis=1)
    return speed/distance


def dist(r1, r2):
    '''
    Parameters r1, r2: np.array of shape r = [x,y]
    Returns absolute value of the distance of connection vector
    '''
    return np.sqrt(np.sum((r1-r2)**2))


def crash_detect(r):
    '''
    Detect crashes of celestial bodies

    Parameters
    ----------
    r : np.array((3, 2))
        x and y positions of the objects Earth (index 0), Moon (index 1)
        and Death Star (index 2) for one time.

    Returns
    -------
    crash : bool
        did a crash happen? True or False.

    '''
    crash = False
    [r1, r2, r3] = r
    if dist(r1, [0,0]) < (radius['Sun']+radius['Earth']):
        print('\nEarth crashed into the Sun!')
        crash = True
    elif dist(r1, r2) < (radius['Moon']+radius['Earth']):
        print('\nMoon crashed into the Earth!')
        crash = True
    elif dist(r1, r3) < (radius['DS']+radius['Earth']):
        print('\nDeathstar crashed into the Earth!')
        crash = True
    elif dist(r3, np.array([0,0])) < (radius['DS']+radius['Sun']):
        print('\nDeathstar crashed into the Sun!')
        crash = True
        
    return crash


# calculate the total force
# r: input array [[x_Earth,y_Earth],[x_Moon,y_Moon],[x_DS,y_DS]]
def F_total(r, m_earth=mass['Earth'], m_DS=mass['DS_A'],
            sun=True, moon=False, DS=False, earth_fixed=False):
    '''
    Calculate the total force acting between the bodies due to
    Newton's law of Gravity

    Parameters
    ----------
    r : np.array((3, 2))
        x and y positions of the objects Earth (index 0), Moon (index 1)
        and Death Star (index 2) for one time.
    m_earth : float, optional,
        mass earth in sun masses, default is mass['Earth'].
    m_DS : float, optional
        mass Death Star in sun masses. The default is mass['DS_A'].
    sun : bool, optional
        Toggles forces of the sun on or off. The default is True.
    moon : bool, optional
        Toggles Moon on or off. The default is False.
    DS : bool, optional
        Toggles Death Star on or off. The default is False.
    earth_fixed : bool, optional
        Toggles whether Earth is held fixed on or off. The default is False.

    Returns
    -------
    F_tot : np.array((3, 2))
        F_x and F_Y total forces acting on Earth (index 0), Moon (index 1)
        and Death Star (index 2).

    '''
    
    # masses
    m_sun = mass['Sun']
    m_moon = mass['Moon']
    
    # radii of objects
    r_earth = r[0]
    r_moon = r[1]
    r_DS = r[2]
    
    # deactivate sun if only orbits around Earth shall be observed
    sunfactor = (1 if sun else 0) # ;)
    
    # deactivate forces acting on earth if earth should be held fixed
    earthfactor = (0 if earth_fixed else 1)
    
    # note: F_x_y means force points from y to x
    if moon:
        # simulate moon
        F_sun_moon = F_gravity(r_moon, m_moon, m_sun) * sunfactor
        F_earth_moon = F_gravity(r_moon-r_earth, m_moon, m_earth)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun) * sunfactor
            F_earth_DS = F_gravity(r_DS-r_earth, m_DS, m_earth)
            F_moon_DS = F_gravity(r_DS-r_moon, m_DS, m_moon)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = F_moon_DS = np.zeros(2)
    
    else:
        #no interactiosn with moon
        F_sun_moon = F_earth_moon = F_moon_DS = np.zeros(2)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun) * sunfactor
            F_earth_DS = F_gravity(r_DS-r_earth, m_DS, m_earth)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = np.zeros(2)
            
    # always: Earth-Sun
    F_sun_earth = F_gravity(r_earth, m_earth, m_sun) * sunfactor
    
    # total force
    F_tot = np.array([F_sun_earth - F_earth_moon - F_earth_DS,
                      F_sun_moon + F_earth_moon - F_moon_DS,
                      F_sun_DS + F_earth_DS + F_moon_DS])
    # account for fixed earht (=setting all forces acting on earth to zero)
    F_tot[0] *= earthfactor
    
    return F_tot
    


# main algorithm: integrate the orbits
def integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'], 
                     distance_DS=distance['DS'], _speed_DS=speed_DS,
                     dt=0.1, t_max=160, sun=True, moon=False, DS=False,
                     earth_fixed=False):
    '''
    Integrate Equations of motions with Velocity Verlet to calculate orbits

    Parameters
    ----------
    m_earth : float, optional,
        mass earth in sun masses, default is mass['Earth'].
    m_DS : float, optional
        mass Death Star in sun masses. The default is mass['DS_A'].
    distance_DS : float, optional
        distance Earth-DS in AU. The default is distance['DS'].
    _speed_DS : float, optional
        orbit speed of DS around Earth in AU/day. The default is speed_DS.
    dt : float, optional
        integration time step in days. The default is 0.1.
    t_max : float, optional
        max integration time in days. The default is 160.
    sun : bool, optional
        Toggles forces of the sun on or off. The default is True.
    moon : bool, optional
        Toggles Moon on or off. The default is False.
    DS : bool, optional
        Toggles Death Star on or off. The default is False.
    earth_fixed : bool, optional
        Toggles whether Earth is held fixed on or off. The default is False.

    Returns
    -------
    time : np.array((timesteps))
    rt : np.array((timesteps, 3, 2))
        x and y positions of the objects Earth (index 0), Moon (index 1)
        and Death Star (index 2) for all time steps.
    vt : np.array((timesteps, 3, 2))
        v_x and v_y velocities of the objects Earth (index 0), Moon (index 1)
        and Death Star (index 2) for all time steps.

    '''
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)
    
    # masses
    m_moon = mass['Moon']
    
    # speeds from global variables
    _speed_earth = speed_earth * (1 if sun else 0) # when sun neglected, see earth as center
    _speed_moon = speed_moon
    
    rt = np.zeros((nsteps, 3, 2)) # [[x_ea,y_ea],[x_mo,y_mo],[x_DS,y_DS]] for every time step
    vt = np.zeros_like(rt)
    
    # initialising earth
    rt[0,0,:] = initial_position(distance['Earth'])
    vt[0,0,:] = np.array([0, _speed_earth])
    
    # initialising moon
    rt[0,1,:] = initial_position(distance['Earth']) + np.array([0,distance['Moon']])
    vt[0,1,:] = np.array([- _speed_moon, _speed_earth])
    
    # initialising Death Star
    rt[0,2,:] = initial_position(distance['Earth']) + np.array([0,-distance_DS])
    vt[0,2,:] = np.array([1*_speed_DS, _speed_earth])
    
    # print(np.sqrt(np.sum((vt[0])**2, axis=1)))
    # print()

    # integration verlocity verlet
    Ft = F_total(rt[0], m_earth=m_earth, m_DS=m_DS, sun=sun, moon=moon, DS=DS,
                 earth_fixed=earth_fixed)
    
    for i in tqdm.tqdm(range(nsteps-1)):
        # print(vt[i,0])
        # print(dist(rt[i,2], np.array([0,0])), Ft[2])
        m = np.array([[m_earth, m_earth],
                     [m_moon, m_moon],
                     [m_DS, m_DS]])
        vhalf = vt[i] + 0.5 * dt * Ft / m
        rt[i+1] = rt[i] + dt * vhalf
    
        # new force
        Ft = F_total(rt[i+1], m_earth=m_earth, m_DS=m_DS, moon=moon, DS=DS,
                     sun=sun, earth_fixed=earth_fixed)
        # print(Ft[1])
        vt[i+1] = vhalf + 0.5 * dt * Ft / m
        # print(np.sqrt(np.sum((vt[i+1])**2, axis=1)))
        
        # crash detection: important to stop, otherwise F explodes
        if crash_detect(rt[i+1]):
            print('STOP SIMULATION')
            print('CELESTIAL CATASTROPHE, PEOPLE DIED!')
            # set all remaining r to the current r[i]
            rt = np.where(rt==np.zeros((3,2)), rt[i+1], rt)
            break

    return time, rt, vt


def orbit_time(r, t, neglect_first=100, eps=1e-4):
    '''
    Calculate the period it takes an object to complete an orbit

    Parameters
    ----------
    r : np.array((timesteps, 1, 2))
        x and y positions of the desired object for all time steps.
        use slicing to select Earth (0), Moon(1) or the DS(2)
    t : np.array((timesteps))
    neglect_first : int, optional
        How many entries in the r-array shall be neglected. The default is 100.
    eps : float, optional
        threshold for how close the Earth has to come back to the initial
        position for the period to be seen as completed. The default is 1e-4.

    Returns
    -------
    period_time: float
        period time of Earth. =0 when time could not be calculated

    '''
    r0 = r[0]
    period_time = 0
    for index, ri in enumerate(r[neglect_first:]):
        if dist(r0, ri) < eps:
            print('Orbit time successfully calculated.')
            period_time = time[neglect_first + index]
            break
    else:
        print('Orbit time could NOT be calculated.')
    return period_time


def stretch_distance(rt, index=1, alpha=30):
    '''
    Stretch distance of objects with respect to Earth (better visibility)

    Parameters
    ----------
    rt : np.array((timesteps, 3, 2))
        x and y positions of the objects Earth (index 0), Moon (index 1)
        and Death Star (index 2) for all time steps.
    index : int, optional
        1 for Moon, 2 for Death Star. The default is 1.
    alpha : float, optional
        value the distance will be stretched with. The default is 30.

    Returns
    -------
    rt : np.array((timesteps, 3, 2))
        new x and y positions with stretched distances
    str
        String for documentation purposes.

    '''
    r1 = rt[:,0]
    r2 = rt[:,index]
    rt[:,index] = r1 + alpha * (r2 - r1)
    return rt, '(stretched)'
        


# =============================================================================
# 3. Calculating Orbits
# =============================================================================

# initialise parameters of orbit calculation
dt=1e-3
t_max=60
sun=True
DS=True
moon=True
earth_fixed=False
moon_stretch = ''
DS_stretch = ''

# calculate orbits
time, r, v = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'],
                                 sun=sun, moon=moon, DS=DS, dt=dt, t_max=t_max,
                                 earth_fixed=earth_fixed)

# stretch coordinates if necessary    
r, moon_stretch = stretch_distance(r, 1)
r, DS_stretch = stretch_distance(r, 2, 200)


# plot
if DS:
    plt.plot(r[:,2,0], r[:,2,1], c='black')
    plt.plot(r[0,2,0], r[0,2,1], marker='o', c='black', markersize=5,
             label=f'DS {DS_stretch}')
    
if moon:
    plt.plot(r[:,1,0], r[:,1,1], c='grey')
    plt.plot(r[0,1,0], r[0,1,1], marker='o', c='grey', markersize=5, 
             label=f'Moon {moon_stretch}')

plt.plot(r[:,0,0], r[:,0,1], c='b')
plt.plot(r[0,0,0], r[0,0,1], marker='o', c='b', markersize=8, label='Earth')

    
if sun:
    plt.plot(0,0, marker='o', markersize=20, color='yellow')
# plt.plot(0,0, marker='o', markersize=20, color='yellow')


plt.title(f'DS_A and Moon orbiting around Earth ({t_max} days)')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc=2)
plt.axis('equal')
# plt.savefig('earth_moon_DS_A.png', dpi=300, bbox_inches='tight')
plt.show()


# Calculation of orbit times:
    # note that t_max should be more than 366 in order for the calculation to
    # work
    
# print(orbit_time(r[:,2], time, eps=4e-4))

# =============================================================================
# 4. Compare omega
# =============================================================================

# change t_max for different time scales
t_max = 10

dt=1e-1
time1, r1, v1 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'],
                              moon=False, DS=False, dt=dt, t_max=t_max)

dt=1e-1
time2, r2, v2 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_B'],
                              moon=True, DS=False, dt=dt, t_max=t_max)

dt=1e-3
time3, r3, v3 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_B'],
                              moon=False, DS=True, dt=dt, t_max=t_max)


o1 = omega(v1[:,0], r1[:,0])
o2 = omega(v2[:,0], r2[:,0])
o3 = omega(v3[:,0], r3[:,0])
plt.plot(time3,o3, label='Earth, DS_B', c='grey')
plt.plot(time1,o1, label='Earth')
plt.plot(time2,o2, label='Earth, Moon')
plt.legend()
plt.xlabel('t [days]')
plt.ylabel('$\omega$ [AU/day]')
plt.title(f'Angular Velocity of Earth ({t_max} days)')
# plt.savefig('omega_earth_1year.png', dpi=300, bbox_inches='tight')
plt.show()

#
# IMPORTANT:
# For more plots, see the Notebook orbits_earth.ipynb    
#



