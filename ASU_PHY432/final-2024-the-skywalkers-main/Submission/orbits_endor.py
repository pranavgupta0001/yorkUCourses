# PHY432: Final Project
# Team: The Skywalkers
# Members: Simon Tebeck, Pranav Gupta
# April 2024

# Objective 2: Orbits Endor

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
mass_sun = 1.9885E+30 # kg, not really necessary here
mass_earth = 5.9724E+24 # kg, not really necessary here
mass_moon = 7.346E+22   # kg we are not sure if Endor has own moons as well
                        # since Endor itself is technically already the moon
                        # of the BigBrother planet (also called Endor, but
                        # we call it BigBrother for clarification).
                        # But since the 'infrastructure' to calculate the
                        # orbits of a moon is already in place, we can leave
                        # it here and pretend that there is one if we want
mass_endor = 7.52E+23 # kg
mass_bb = 2.01E+27 # kg


# Astronomical units
# DS_A: Death Star (light estimation)
# DS_B: Death Star (heavy estimation)
mass = {'BB': 1.,
        'Earth': mass_earth/mass_bb,
        'Moon': mass_moon/mass_bb,
        'Endor': mass_endor/mass_bb,
        'DS_A': 1e18/mass_bb,
        'DS_B': 2.8e23/mass_bb,
        }


# speeds in AU/day (Earth-day)
# for endor: use pre-determined speed for circular orbit: 103.04 km/s
speed_endor = 103.04 * 3600*24 / au # speed endor around bb
# for moon: speed at apogee: 0.966 km/s
speed_moon =  0.966 * 3600*24 / au # speed moon around endor
# for DS: number that leads to stable orbits
speed_DS = 44.56 * 3600*24 / au # speed DS around endor


# distance from the respective orbit center in AU
distance = {'Endor': 1.25e7 / au, # e7 is in km!!
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

# print('Speed of Endor around BB for circular orbit:')
print(np.sqrt(G_gravity*mass['Endor']/distance['DS'])*au/(3600*24))


# =============================================================================
# 2. Functions for the orbit calculations (Endor System) (for Solar System
# see other file in Submission)
# =============================================================================

# note: Some of the following functions were inspired by HW07, code templates
# offered by Oliver Beckstein, modified by Simon Tebeck



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
        x and y positions of the objects endor (index 0), Moon (index 1)
        and Death Star (index 2) for one time.

    Returns
    -------
    crash : bool
        did a crash happen? True or False.

    '''
    crash = False
    [r1, r2, r3] = r
    if dist(r1, [0,0]) < (radius['BB']+radius['Endor']):
        print('\nEndor crashed into the Big Brother!')
        crash = True
    elif dist(r1, r2) < (radius['Moon']+radius['Endor']):
        print('\nMoon crashed into the Endor!')
        crash = True
    elif dist(r1, r3) < (radius['DS']+radius['Endor']):
        print('\nDeathstar crashed into the Endor!')
        crash = True
    elif dist(r3, np.array([0,0])) < (radius['DS']+radius['BB']):
        print('\nDeathstar crashed into the Big Brother!')
        crash = True
        
    return crash


# calculate the total force
# r: input array [[x_endor,y_endor],[x_Moon,y_Moon],[x_DS,y_DS]]
def F_total(r, m_endor=mass['Endor'], m_DS=mass['DS_A'],
            BB=True, moon=False, DS=False, endor_fixed=False):
    '''
    Calculate the total force acting between the bodies due to
    Newton's law of Gravity

    Parameters
    ----------
    r : np.array((3, 2))
        x and y positions of the objects endor (index 0), Moon (index 1)
        and Death Star (index 2) for one time.
    m_endor : float, optional,
        mass endor in BB masses, default is mass['Endor'].
    m_DS : float, optional
        mass Death Star in BB masses. The default is mass['DS_A'].
    BB : bool, optional
        Toggles forces of the BB on or off. The default is True.
    moon : bool, optional
        Toggles Moon on or off. The default is False.
    DS : bool, optional
        Toggles Death Star on or off. The default is False.
    endor_fixed : bool, optional
        Toggles whether Endor is held fixed on or off. The default is False.

    Returns
    -------
    F_tot : np.array((3, 2))
        F_x and F_Y total forces acting on Endor (index 0), Moon (index 1)
        and Death Star (index 2).

    '''
    
    # masses
    m_BB = mass['BB']
    m_moon = mass['Moon']
    
    # radii of objects
    r_endor = r[0]
    r_moon = r[1]
    r_DS = r[2]
    
    # deactivate BB if only orbits around endor shall be observed
    BBfactor = (1 if BB else 0) # ;)
    
    # deactivate forces acting on endor if endor should be held fixed
    endorfactor = (0 if endor_fixed else 1)
    
    # note: F_x_y means force points from y to x
    if moon:
        # simulate moon
        F_BB_moon = F_gravity(r_moon, m_moon, m_BB) * BBfactor
        F_endor_moon = F_gravity(r_moon-r_endor, m_moon, m_endor)
        
        if DS:
            # simulate DS
            F_BB_DS = F_gravity(r_DS, m_DS, m_BB) * BBfactor
            F_endor_DS = F_gravity(r_DS-r_endor, m_DS, m_endor)
            F_moon_DS = F_gravity(r_DS-r_moon, m_DS, m_moon)
        else:
            # no interactions with DS
            F_BB_DS = F_endor_DS = F_moon_DS = np.zeros(2)
    
    else:
        #no interactiosn with moon
        F_BB_moon = F_endor_moon = F_moon_DS = np.zeros(2)
        
        if DS:
            # simulate DS
            F_BB_DS = F_gravity(r_DS, m_DS, m_BB) * BBfactor
            F_endor_DS = F_gravity(r_DS-r_endor, m_DS, m_endor)
        else:
            # no interactions with DS
            F_BB_DS = F_endor_DS = np.zeros(2)
            
    # always: endor-BB
    F_BB_endor = F_gravity(r_endor, m_endor, m_BB) * BBfactor
    
    # total force
    F_tot = np.array([F_BB_endor - F_endor_moon - F_endor_DS,
                      F_BB_moon + F_endor_moon - F_moon_DS,
                      F_BB_DS + F_endor_DS + F_moon_DS])
    # account for fixed earht (=setting all forces acting on endor to zero)
    F_tot[0] *= endorfactor
    
    return F_tot
    


# main algorithm: integrate the orbits
def integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_A'], 
                     distance_DS=distance['DS'], _speed_DS=speed_DS,
                     dt=0.1, t_max=160, BB=True, moon=False, DS=False,
                     endor_fixed=False):
    '''
    Integrate Equations of motions with Velocity Verlet to calculate orbits

    Parameters
    ----------
    m_endor : float, optional,
        mass endor in BB masses, default is mass['Endor'].
    m_DS : float, optional
        mass Death Star in BB masses. The default is mass['DS_A'].
    distance_DS : float, optional
        distance endor-DS in AU. The default is distance['DS'].
    _speed_DS : float, optional
        orbit speed of DS around endor in AU/day. The default is speed_DS.
    dt : float, optional
        integration time step in days. The default is 0.1.
    t_max : float, optional
        max integration time in days. The default is 160.
    BB : bool, optional
        Toggles forces of the BB on or off. The default is True.
    moon : bool, optional
        Toggles Moon on or off. The default is False.
    DS : bool, optional
        Toggles Death Star on or off. The default is False.
    endor_fixed : bool, optional
        Toggles whether endor is held fixed on or off. The default is False.

    Returns
    -------
    time : np.array((timesteps))
    rt : np.array((timesteps, 3, 2))
        x and y positions of the objects endor (index 0), Moon (index 1)
        and Death Star (index 2) for all time steps.
    vt : np.array((timesteps, 3, 2))
        v_x and v_y velocities of the objects endor (index 0), Moon (index 1)
        and Death Star (index 2) for all time steps.

    '''
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)
    
    # masses
    m_moon = mass['Moon']
    
    # speeds from global variables
    _speed_endor = speed_endor * (1 if BB else 0) # when BB neglected, see endor as center
    _speed_moon = speed_moon
    
    rt = np.zeros((nsteps, 3, 2)) # [[x_ea,y_ea],[x_mo,y_mo],[x_DS,y_DS]] for every time step
    vt = np.zeros_like(rt)
    
    # initialising endor
    rt[0,0,:] = np.array([distance['Endor'], 0])
    vt[0,0,:] = np.array([0, _speed_endor])
    
    # initialising moon
    rt[0,1,:] = np.array([distance['Endor'], 0]) + np.array([0,distance['Moon']])
    vt[0,1,:] = np.array([- _speed_moon, _speed_endor])
    
    # initialising Death Star
    rt[0,2,:] = np.array([distance['Endor'], 0]) + np.array([0,-distance_DS])
    vt[0,2,:] = np.array([1*_speed_DS, _speed_endor])
    
    # print(np.sqrt(np.sum((vt[0])**2, axis=1)))
    # print()

    # integration verlocity verlet
    Ft = F_total(rt[0], m_endor=m_endor, m_DS=m_DS, BB=BB, moon=moon, DS=DS,
                 endor_fixed=endor_fixed)
    
    for i in tqdm.tqdm(range(nsteps-1)):
        # print(vt[i,0])
        # print(dist(rt[i,2], np.array([0,0])), Ft[2])
        m = np.array([[m_endor, m_endor],
                     [m_moon, m_moon],
                     [m_DS, m_DS]])
        vhalf = vt[i] + 0.5 * dt * Ft / m
        rt[i+1] = rt[i] + dt * vhalf
    
        # new force
        Ft = F_total(rt[i+1], m_endor=m_endor, m_DS=m_DS, moon=moon, DS=DS,
                     BB=BB, endor_fixed=endor_fixed)
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
    Calculate the period it takes endor to complete an orbit

    Parameters
    ----------
    r : np.array((timesteps, 1, 2))
        x and y positions of the desired object for all time steps.
        use slicing to select Endor (0), Moon(1) or the DS(2)
    t : np.array((timesteps))
    neglect_first : int, optional
        How many entries in the r-array shall be neglected. The default is 100.
    eps : float, optional
        threshold for how close the endor has to come back to the initial
        position for the period to be seen as completed. The default is 1e-4.

    Returns
    -------
    period_time: float
        period time of endor. =0 when time could not be calculated

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
    Stretch distance of objects with respect to endor (better visibility)

    Parameters
    ----------
    rt : np.array((timesteps, 3, 2))
        x and y positions of the objects endor (index 0), Moon (index 1)
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
t_max=4
BB=True
DS=True
moon=False
endor_fixed=False
moon_stretch = ''
DS_stretch = ''

# calculate orbits
time, r, v = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                                 BB=BB, moon=moon, DS=DS, dt=dt, t_max=t_max,
                                 endor_fixed=endor_fixed)

# stretch coordinates if necessary    
r, moon_stretch = stretch_distance(r, 1)
r, DS_stretch = stretch_distance(r, 2, 100)


# plot
if DS:
    plt.plot(r[:,2,0], r[:,2,1], c='black')
    plt.plot(r[0,2,0], r[0,2,1], marker='o', c='black', markersize=5,
             label=f'DS {DS_stretch}')
    
if moon:
    plt.plot(r[:,1,0], r[:,1,1], c='grey')
    plt.plot(r[0,1,0], r[0,1,1], marker='o', c='grey', markersize=5, 
             label=f'Moon {moon_stretch}')

plt.plot(r[:,0,0], r[:,0,1], c='g')
plt.plot(r[0,0,0], r[0,0,1], marker='o', c='g', markersize=8, label='Endor')

    
if BB:
    plt.plot(0,0, marker='o', markersize=30, color='brown')
# plt.plot(0,0, marker='o', markersize=30, color='brown')


plt.title(f'DS_B orbiting around Endor ({t_max} days)')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc=2)
plt.axis('equal')
# plt.savefig(f'endor_DS_B_{t_max}_days.png', dpi=300, bbox_inches='tight')
plt.show()



# make comparison plot

# initialise parameters of orbit calculation
dt=1e-3
t_max=12
BB=True
DS=True
moon=False
endor_fixed=False
moon_stretch = ''
DS_stretch = ''

# calculate orbits
time, r, v = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                                 BB=BB, moon=moon, DS=False, dt=dt, t_max=t_max,
                                 endor_fixed=endor_fixed)
time2, r2, v2 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_A'],
                                 BB=BB, moon=moon, DS=DS, dt=dt, t_max=t_max,
                                 endor_fixed=endor_fixed)
time3, r3, v3 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                                 BB=BB, moon=moon, DS=DS, dt=dt, t_max=t_max,
                                 endor_fixed=endor_fixed)

# stretch coordinates if necessary    
# r, moon_stretch = stretch_distance(r, 1)
# r, DS_stretch = stretch_distance(r, 2, 100)


# plot

plt.plot(r[:,0,0], r[:,0,1], linewidth=5, c='g')
plt.plot(r[0,0,0], r[0,0,1], marker='o', c='g', markersize=8, label='Endor')
plt.plot(r2[:,0,0], r2[:,0,1], c='y')
plt.plot(r2[0,0,0], r2[0,0,1], marker='o', c='y', markersize=8, label='Endor, DS_A')
plt.plot(r3[:,0,0], r3[:,0,1], c='r')
plt.plot(r3[0,0,0], r3[0,0,1], marker='o', c='r', markersize=8, label='Endor, DS_B')

    
if BB:
    plt.plot(0,0, marker='o', markersize=30, color='brown')
# plt.plot(0,0, marker='o', markersize=30, color='brown')


plt.title(f'Endors (pertubated) Orbits ({t_max} days)')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc=2)
plt.axis('equal')
# plt.savefig(f'endor_orbits_{t_max}_days.png', dpi=300, bbox_inches='tight')
plt.show()


# Calculation of orbit times:
    # note that t_max should be more than 10 in order for the calculation to
    # work
print('Orbit period Endor normal:') 
print(orbit_time(r[:,0], time, eps=4e-4))
print('Orbit period Endor DS_A:') 
print(orbit_time(r2[:,0], time, eps=4e-4))
print('Orbit period Endor DS_B:') 
print(orbit_time(r3[:,0], time, eps=4e-4))

# =============================================================================
# 4. Compare omega
# =============================================================================

# change t_max for different time scales

# DS_A

t_max = 10

dt=1e-2
time1, r1, v1 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_A'],
                              moon=False, DS=False, dt=dt, t_max=t_max)

dt=1e-4
time3, r3, v3 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_A'],
                              moon=False, DS=True, dt=dt, t_max=t_max)

o1 = omega(v1[:,0], r1[:,0])
o3 = omega(v3[:,0], r3[:,0])
plt.plot(time3,o3, label='Endor, DS_A', c='grey')
plt.plot(time1,o1, label='Endor')
plt.legend()
plt.xlabel('t [days]')
plt.ylabel('$\omega$ [AU/day]')
plt.title(f'Angular Velocity of Endor ({t_max} days)')
# plt.savefig(f'omega_endor_DS_A_{t_max}_days.png', dpi=300, bbox_inches='tight')
plt.show()



# DS_B

t_max = 10

dt=1e-2
time1, r1, v1 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                              moon=False, DS=False, dt=dt, t_max=t_max)

dt=1e-4
time3, r3, v3 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                              moon=False, DS=True, dt=dt, t_max=t_max)

o1 = omega(v1[:,0], r1[:,0])
o3 = omega(v3[:,0], r3[:,0])
plt.plot(time3,o3, label='Endor, DS_B', c='grey')
plt.plot(time1,o1, label='Endor')
plt.legend()
plt.xlabel('t [days]')
plt.ylabel('$\omega$ [AU/day]')
plt.title(f'Angular Velocity of Endor ({t_max} days)')
# plt.savefig(f'omega_endor_DS_B_{t_max}_days.png', dpi=300, bbox_inches='tight')
plt.show()


# DS_B less time

# change t_max for different time scales
t_max = 1e-1

dt=1e-3
time1, r1, v1 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_A'],
                              moon=False, DS=False, dt=dt, t_max=t_max)

# dt=1e-1
# time2, r2, v2 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
#                               moon=True, DS=False, dt=dt, t_max=t_max)

dt=1e-5
time3, r3, v3 = integrate_orbits(m_endor=mass['Endor'], m_DS=mass['DS_B'],
                              moon=False, DS=True, dt=dt, t_max=t_max)


o1 = omega(v1[:,0], r1[:,0])
# o2 = omega(v2[:,0], r2[:,0])
o3 = omega(v3[:,0], r3[:,0])
plt.plot(time3,o3, label='Endor, DS_B', c='grey')
plt.plot(time1,o1, label='Endor')
# plt.plot(time2,o2, label='Endor, Moon')
plt.legend()
plt.xlabel('t [days]')
plt.ylabel('$\omega$ [AU/day]')
plt.title(f'Angular Velocity of Endor ({t_max} days)')
# plt.savefig(f'omega_endor_DS_B_{t_max}_days.png', dpi=300, bbox_inches='tight')
plt.show()



