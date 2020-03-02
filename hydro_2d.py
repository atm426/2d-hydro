import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib import colors
from matplotlib.patches import Circle


# 2d code

class hydro:

    def __init__(self, J):
        # setting up initial conditions and arrays
        self.t = 0.0 ; self.J = J ; self.dt = 0
        self.cfl = 0.7 ; self.t0 = -1
        
        # vector of conserved quantities
        self.u = np.zeros(( 5, J, J ))
        # u[0,:] = density
        # u[1,:] = density * v_x
        # u[2,:] = density * v_y
        # u[3,:] = energy
        # u[4,:] = density * q (passive scalar)
        
        # vector of primitive variables
        self.prim = np.zeros(( 5, J, J ))
        # prim[0,:] = density
        # prim[1,:] = pressure
        # prim[2,:] = v_x
        # prim[3,:] = v_y
        # prim[4,:] = q
        

        self.f = np.zeros(( 5, J, J ))
        # f[0,:] = rho * v_x
        # f[1,:] = rho * v_x^2 + P
        # f[2,:] = rho * v_x * v_y
        # f[3,:] = v_x * (E + P)
        # f[4,:] = density * q * v_x

        self.g = np.zeros(( 5, J, J ))
        # g[0,:] = rho * v_y
        # g[1,:] = rho * v_x * v_y
        # g[2,:] = rho * v_y^2 + P
        # g[3,:] = v_y * (E + P)
        # g[4,:] = density * q * v_y

    def main_fn(self, problem, tf, filename, save_data, load_data, start):
        # main function
        ''' problem: which test problem to do
        tf : final time to run to
        filename : prefix for image names'''

        # remove old images
        if filename is not None:
            os.system('rm /Users/austin/classes/comp_phys/finalproject-atm426/imgs/'+filename+'*.png')
            
        if not load_data:
            start = '0'
        self.test_prob(problem, start, load_data=load_data)
        self.evolve_u(tf, filename, problem, save_data, load_data)
        
        # open new images
        if filename is not None:
            os.system('open /Users/austin/classes/comp_phys/finalproject-atm426/imgs/'+filename+'*.png')

    def evolve_u(self, tf, filename, problem, save_data, load_data):
        # evolves eulers equations with hll flux solver
        # tf = final time
        
        if not load_data:
            self.n = 0 # time step number
        
        # plot initial conditions
        self.plot_hydro(tf, problem, filename)
        k = 0

        while (self.t+self.dt) < tf:
            # save a copy of u_n
            u_n = np.copy(self.u)
            prim_n = np.copy(self.prim)
            flux_n = np.copy(self.f)

            # u(1) 
            self.u[:] = u_n + self.dt*self.hllc()
            
            # save copy of u(1)
            u_1 = np.copy(self.u)
            
            #update primitive variables and fluxes
            self.get_prim(problem, save_data=False)
            self.update_f()

            # u(2)
            self.u[:] = (3/4.)*u_n + (1/4.)*u_1 + (1/4.)*self.dt*self.hllc()
            
            # save copy of u(2)
            u_2 = np.copy(self.u)

            # update primitive variables and fluxes
            self.get_prim(problem, save_data=False)
            self.update_f()

            # u(n+1)
            self.u[:] = (1/3.)*u_n + (2/3.)*u_2 + (2/3.)*self.dt*self.hllc()
            if load_data and k==0:
                self.t = int(self.n)*self.dt
                k = 1
            else:
                self.t += self.dt

            # periodic boundary conditions in y
            #self.u[:, 3, :] = self.u[:, self.u[:,0,:].shape[1]-5, :] 
            #self.u[:, self.u[:,1,:].shape[1]-4, :] = self.u[:, 4, :] 

            # periodic boundary conditions in x
            #self.u[:, :, 3] = self.u[:, :, self.u[:,0,:].shape[1]-5] 
            #self.u[:, :, self.u[:,1,:].shape[1]-4] = self.u[:, :, 4] 

            # reset primitive variables to values at t = t_n
            self.prim = prim_n
            self.f = flux_n
        
            # calculate primitive variables from new u values
            self.get_prim(problem, save_data=save_data)

            # update fluxes in each zone
            self.update_f()
            self.n += 1
            
            

            self.plot_hydro(tf, problem, filename)

            
    def update_f(self):
        # update flux arrays

        self.f[0] = np.copy(self.u[1]) # rho * v_x
        self.f[1] = self.u[1] * self.prim[2] + self.prim[1] # rho*v_x^2 + P
        self.f[2] = self.u[1] * self.prim[3] # rho * v_x * v_y
        self.f[3] = self.prim[2]*( self.u[3] + self.prim[1] ) # (E+P) * v_x
        self.f[4] = self.prim[0]*self.prim[4]*self.prim[2]

        self.g[0] = np.copy(self.u[2]) # rho * v_y
        self.g[1] = np.copy(self.f[2]) # rho * v_x * v_y
        self.g[2] = self.u[2] * self.prim[3] + self.prim[1] # rho*v_y^2 + P
        self.g[3] = self.prim[3]*( self.u[3] + self.prim[1] ) # (E+P) * v_y
        self.g[4] = self.prim[0]*self.prim[4]*self.prim[3]

    def get_prim(self,problem, save_data=False):
        # solve for primitive variables
        
        self.prim[0] = np.copy(self.u[0]) # density
        self.prim[2] = self.u[1]/self.u[0] # v_x 
        self.prim[3] = self.u[2]/self.u[0] # v_y 
        self.prim[1] = (self.u[3]-0.5*self.prim[0]*(self.prim[2]*self.prim[2]+self.prim[3]*self.prim[3]))*(self.gamma-1.) # P
        self.prim[4] = self.u[4]/self.prim[0]
            
        self.cs = np.sqrt( self.gamma * self.prim[1] / self.prim[0] ) # sound speed

        if save_data:
            if self.n % 10 == 0:
                output = open('/Users/austin/classes/comp_phys/finalproject-atm426/data/'+problem+'_data_'+str(self.n).zfill(4)+'.pkl', 'wb')
                pickle.dump(self.prim, output)
                output.close

                print 'saved output file ' + str(self.n)

    def plm(self, i):
        # uses piecewise linear method to extrapolate to cell boundaries
        # if i=0 calculate for x-dir
        # if i=1 calculate for y-dir
        
        if i == 0:
            # along x-dir
            cl = self.prim[:, :, 2:-2] - self.prim[:, :, 1:-3]
            cc = self.prim[:, :, 2:] - self.prim[:, :, :-2]
            cr = self.prim[:, :, 2:-2] - self.prim[:, :, 3:-1]
        
            # left biased values
            cll = cl[:, :, :-1] ; ccl = cc[:, :, :-1] ; crl = cr[:, :, :-1]

            # left biased slopes
            sl = self.minmod(cll, crl, ccl, i)

            # right biased values
            clr = cl[:, :, 1:] ; ccr = cc[:, :, 1:] ; crr = cr[:, :, 1:]
            sr = self.minmod(clr, crr, ccr, i)

        else:
            # along y-dir
            cl = self.prim[:, 2:-2, :] - self.prim[:, 1:-3, :]
            cc = self.prim[:, 2:, :] - self.prim[:, :-2, :]
            cr = self.prim[:, 2:-2, :] - self.prim[:, 3:-1, :]
        
            # left biased values
            cll = cl[:, :-1, :] ; ccl = cc[:, :-1, :] ; crl = cr[:, :-1, :]

            # left biased slopes
            sl = self.minmod(cll, crl, ccl, i)

            # right biased values
            clr = cl[:, 1:, :] ; ccr = cc[:, 1:, :] ; crr = cr[:, 1:, :]
            sr = self.minmod(clr, crr, ccr, i)
        
        return sl, sr

    def minmod(self, x,y,z, i):
        # minmod function for PLM
        ''' x : c_L = (c_i - c_i-1)
        y : c_R = (c_i+1 - c_i)
        z : c_C = (c_i+1 - c_i-1)'''
        
        # these arrays wont change depending on x and y, so only calculate once
        sgn_x = np.array( map(np.sign, x) )
        sgn_y = np.array( map(np.sign, y) )
        sgn_z = np.array( map(np.sign, z) )

        theta = 1.5
        x *= theta
        y *= theta
        z *= 0.5

        # array to hold minimum values
        min_arr = np.zeros(sgn_x.shape)
    
        # if along x-dir
        if i == 0:
            axis_len = x.shape[2]
            for j in xrange(axis_len):
                min_arr[:,j,:] = map(min,abs(x[0,j,:]),
                                     abs(y[0,j,:]),
                                     abs(z[0,j,:-2]))

            # slope array for primitive variables
            p_s = (1/4.)*abs( sgn_x + sgn_y )*( sgn_x + sgn_z[:,:,:-2] )*min_arr

        # if along y-dir
        else:
            axis_len = x.shape[1]
            for j in xrange(axis_len):
                min_arr[:,:,j] = map(min,abs(x[0,:,j]),
                                          abs(y[0,:,j]),
                                          abs(z[0,:-2,j]))

            # slope array for primitive variables
            p_s = (1/4.)*abs( sgn_x + sgn_y )*( sgn_x + sgn_z[:,:-2,:] )*min_arr
                
        return p_s

    def hllc(self):
        # calculate HLLC flux
    
        # doing this twice for x-dir (i=0) and y-dir (i=1) 
        for i in xrange(2):
            
            # along x-dir
            if i == 0:

                # left and right slopes for PLM
                sl, sr = self.plm(i)

                # primitive variables on left and right sides
                prim_l = self.prim[:, :, 2:-3] + 0.5*sl
                prim_r = self.prim[:, :, 3:-2] - 0.5*sr

                # conserved quantities on left and right sides
                # left side
                ul = np.zeros_like(prim_l)
                ul[0] = np.copy(prim_l[0]) # rho
                ul[1] = prim_l[0]*prim_l[2] # rho * v_x
                ul[2] = prim_l[0]*prim_l[3] # rho * v_y
                ul[3] = prim_l[1]/(self.gamma - 1) + 0.5*prim_l[0]*(prim_l[2]*prim_l[2]+prim_l[3]*prim_l[3]) # E
                ul[4] = prim_l[0]*prim_l[4] # q

                # right side
                ur = np.zeros_like(prim_r)
                ur[0] = np.copy(prim_r[0]) # rho
                ur[1] = prim_r[0]*prim_r[2] # rho * v_x
                ur[2] = prim_r[0]*prim_r[3] # rho * v_y
                ur[3] = prim_r[1]/(self.gamma - 1) + 0.5*prim_r[0]*(prim_r[2]*prim_r[2]+prim_r[3]*prim_r[3]) # E
                ur[4] = prim_r[0]*prim_r[4] # q


                # fluxes on left and right sides
                # left values for fluxes
                fl = np.zeros_like(prim_l)
                fl[0] = np.copy(ul[1]) # rho * v_x
                fl[1] = ul[1] * prim_l[2] + prim_l[1] # rho*v_x^2 + P
                fl[2] = ul[1] * prim_l[3] # rho * v_x * v_y
                fl[3] = prim_l[2] * ( ul[3] + prim_l[1] ) # (E + P) * v_x
                fl[4] = prim_l[0] * prim_l[4] * prim_l[2]
                # right values for fluxes
                fr = np.zeros_like(prim_r)
                fr[0] = np.copy(ur[1]) # rho * v_x
                fr[1] = ur[1] * prim_r[2] + prim_r[1] # rho*v_x^2 + P
                fr[2] = ur[1] * prim_r[3] # rho * v_x * v_y
                fr[3] = prim_r[2] * ( ur[3] + prim_r[1] ) # (E + P) * v_x
                fr[4] = prim_r[0] * prim_r[4] * prim_r[2]

                # sound speed on left and right side
                csl = np.sqrt(self.gamma * prim_l[1]/prim_l[0])
                csr = np.sqrt(self.gamma * prim_r[1]/prim_r[0])

                # lambda plus and lambda minus for left and right side
                lam_pl = prim_l[2] + csl ; lam_ml = prim_l[2] - csl
                lam_pr = prim_r[2] + csr ; lam_mr = prim_r[2] - csr

                # arrays for lambda plus and minus
                lam_pr_arr = np.zeros(lam_pr.shape[0]) 
                lam_ml_arr = np.zeros(lam_ml.shape[0])
                for j in xrange(lam_pr.shape[0]):
                    lam_pr_arr[j] = min(np.array(map( min, 
                                                      lam_pl[j,:], 
                                                      lam_pr[j,:] )))
                    lam_ml_arr[j] = min(np.array(map( min, 
                                                      lam_ml[j,:], 
                                                      lam_mr[j,:] )))
            
                # lambda star
                denom = (prim_l[0]*csl + prim_r[0]*csr)
                lam_s = (prim_l[1] - prim_r[1] + prim_l[0]*csl*prim_l[2] + prim_r[0]*csr*prim_r[2])/denom

                # alpha plus and alpha minus
                zero_arr = np.zeros(lam_pl.shape[0])
                for j in xrange(lam_pl.shape[0]):
                    ap = np.array( map( max, zero_arr, lam_pl[j,:], lam_pr_arr ) )
                    am = np.array( map( max, zero_arr, -lam_ml_arr, -lam_mr[j,:] ) )

                # timestep size from minimum wave speeds
                dt_arr = np.zeros(lam_pl.shape[0])
                for j in xrange(lam_pl.shape[0]):
                    dt_arr[j] = self.cfl*(self.dx / max( map( max, 
                                                              lam_pl[j,:], 
                                                              lam_pr_arr, 
                                                              -lam_ml_arr, 
                                                              -lam_mr[j,:], 
                                                              lam_s[j,:] )) )
        
                self.dt = dt_arr[ dt_arr>0 ].min()
                                            
                # conserved quantities in left and right starred regions
                u_sr = np.zeros_like(prim_r) ; u_sl = np.zeros_like(prim_l)
                # right side
                right_const = prim_r[0]*(csr/(prim_r[2]+csr-lam_s)) 
                u_sr[0] = np.copy(right_const)
                u_sr[1] = right_const*lam_s
                u_sr[2] = right_const*prim_r[3]
                u_sr[3] = right_const*(ur[3]/prim_r[0] + (lam_s-prim_r[2])*(lam_s+(prim_r[1]/(prim_r[0]*csr))))
                u_sr[4] = right_const * prim_r[4]
                # left side
                left_const = prim_l[0]*(-csl/(prim_l[2]-csl-lam_s)) 
                u_sl[0] = np.copy(left_const)
                u_sl[1] = left_const*lam_s
                u_sl[2] = left_const*prim_l[3]
                u_sl[3] = left_const*(ul[3]/prim_l[0] + (lam_s-prim_l[2])*(lam_s+(prim_l[1]/(-prim_l[0]*csl))))
                u_sl[4] = left_const * prim_l[4]
                # fluxes in left and right starred regions
                # right side
                f_sr = fr - lam_pr*(ur - u_sr)
                # left side
                f_sl = fl - lam_ml*(ul - u_sl)
        
                f_s = np.zeros_like(f_sr)

                # flux in starred region based on wave speeds
                for k in xrange(f_s.shape[0]):
                    f_s[k][0 <= lam_ml] = fl[k][0 <= lam_ml]
                    f_s[k][(lam_ml<=0)*(lam_s>=0)] = f_sl[k][(lam_ml<=0)*(lam_s>=0)]
                    f_s[k][(lam_pr>=0)*(lam_s<=0)] = f_sr[k][(lam_pr>=0)*(lam_s<=0)]
                    f_s[k][0 >= lam_pr] = fr[k][0 >= lam_pr]

            else:
                # along y-dir
                
                # left and right slopes for PLM
                sl, sr = self.plm(i)

                # primitive variables on left and right sides
                prim_l = self.prim[:, 2:-3, :] + 0.5*sl
                prim_r = self.prim[:, 3:-2, :] - 0.5*sr

                # conserved quantities on left and right sides
                # left side
                ul = np.zeros_like(prim_l)
                ul[0] = np.copy(prim_l[0]) # rho
                ul[1] = prim_l[0]*prim_l[2] # rho * v_x
                ul[2] = prim_l[0]*prim_l[3] # rho * v_y
                ul[3] = prim_l[1]/(self.gamma - 1) + 0.5*prim_l[0]*(prim_l[2]*prim_l[2]+prim_l[3]*prim_l[3]) # E
                ul[4] = prim_l[0]*prim_l[4] # q
                # right side
                ur = np.zeros_like(prim_r)
                ur[0] = np.copy(prim_r[0]) # rho
                ur[1] = prim_r[0]*prim_r[2] # rho * v_x
                ur[2] = prim_r[0]*prim_r[3] # rho * v_y
                ur[3] = prim_r[1]/(self.gamma - 1) + 0.5*prim_r[0]*(prim_r[2]*prim_r[2]+prim_r[3]*prim_r[3]) # E
                ur[4] = prim_r[0]*prim_r[4] # q

                # fluxes on left and right sides
                # left values for fluxes
                gl = np.zeros_like(prim_l)
                gl[0] = np.copy(ul[2]) # rho * v_y
                gl[1] = ul[1] * prim_l[3] # rho * v_x * v_y
                gl[2] = ul[2] * prim_l[3] + prim_l[1] # rho * v_y^2 + P
                gl[3] = prim_l[3] * ( ul[3] + prim_l[1] ) # (E + P) * v_y
                gl[4] = prim_l[0]*prim_l[4]*prim_l[3]
                # right values for fluxes
                gr = np.zeros_like(prim_r)
                gr[0] = np.copy(ur[2]) # rho * v_y
                gr[1] = ur[1] * prim_r[3] # rho * v_x * v_y
                gr[2] = ur[2] * prim_r[3] + prim_r[1] # rho * v_y^2 + P
                gr[3] = prim_r[3] * ( ur[3] + prim_r[1] ) # (E + P) * v_y
                gr[4] = prim_r[0]*prim_r[4]*prim_r[3]

                # sound speed on left and right side
                csl = np.sqrt(self.gamma * prim_l[1]/prim_l[0])
                csr = np.sqrt(self.gamma * prim_r[1]/prim_r[0])

                # lambda plus and lambda minus for left and right side
                lam_pl = prim_l[3] + csl ; lam_ml = prim_l[3] - csl
                lam_pr = prim_r[3] + csr ; lam_mr = prim_r[3] - csr

                
                lam_pr_arr = np.zeros(lam_pr.shape[1]) 
                lam_ml_arr = np.zeros(lam_ml.shape[1])
                for j in xrange(lam_pr.shape[1]):
                    lam_pr_arr[j] = min(np.array(map( min, 
                                                      lam_pl[:,j], 
                                                      lam_pr[:,j] )))
                    lam_ml_arr[j] = min(np.array(map( min, 
                                                      lam_ml[:,j], 
                                                      lam_mr[:,j] )))

                # lambda star
                denom = (prim_l[0]*csl + prim_r[0]*csr)                
                lam_s = (prim_l[1] - prim_r[1] + prim_l[0]*csl*prim_l[3] + prim_r[0]*csr*prim_r[3])/denom
                
                # alpha plus and alpha minus
                zero_arr = np.zeros(lam_pl.shape[1])
                for j in xrange(lam_pl.shape[1]):
                    ap = np.array( map( max, zero_arr, lam_pl[:,j], lam_pr_arr ) )
                    am = np.array( map( max, zero_arr, -lam_ml_arr, -lam_mr[:,j] ) )

                # timestep size
                dt_arr = np.zeros(lam_pl.shape[1])
                for j in xrange(lam_pl.shape[1]):
                    dt_arr[j] = self.cfl*(self.dx / max( map( max, 
                                                              lam_pl[:,j], 
                                                              lam_pr_arr, 
                                                              -lam_ml_arr, 
                                                              -lam_mr[:,j], 
                                                              lam_s[:,j] )) )
        
                self.dt = dt_arr[ dt_arr>0 ].min()
                                            
                # conserved quantities in left and right starred regions
                u_sr = np.zeros_like(prim_r) ; u_sl = np.zeros_like(prim_l)
                # right side
                right_const = prim_r[0]*(csr/(prim_r[3]+csr-lam_s)) 
                u_sr[0] = np.copy(right_const)
                u_sr[1] = right_const*prim_r[2]
                u_sr[2] = right_const*lam_s
                u_sr[3] = right_const*(ur[3]/prim_r[0] + (lam_s-prim_r[3])*(lam_s+(prim_r[1]/(prim_r[0]*csr))))
                u_sr[4] = right_const*prim_r[4]
                # left side
                left_const = prim_l[0]*(-csl/(prim_l[3]-csl-lam_s)) 
                u_sl[0] = np.copy(left_const)
                u_sl[1] = left_const*prim_l[2]
                u_sl[2] = left_const*lam_s
                u_sl[3] = left_const*(ul[3]/prim_l[0] + (lam_s-prim_l[3])*(lam_s+(prim_l[1]/(-prim_l[0]*csl))))
                u_sl[4] = left_const*prim_l[4]

                # fluxes in left and right starred regions
                # right side
                g_sr = gr - lam_pr*(ur - u_sr)
                # left side
                g_sl = gl - lam_ml*(ul - u_sl)
        
                g_s = np.zeros_like(g_sr)

                # fluxes in starred regions based on wave speeds
                for k in xrange(g_s.shape[0]):
                    g_s[k][0 <= lam_ml] = gl[k][0 <= lam_ml]
                    g_s[k][(lam_ml<=0)*(lam_s>=0)] = g_sl[k][(lam_ml<=0)*(lam_s>=0)]
                    g_s[k][(lam_pr>=0)*(lam_s<=0)] = g_sr[k][(lam_pr>=0)*(lam_s<=0)]
                    g_s[k][0 >= lam_pr] = gr[k][0 >= lam_pr]

        # array to update conserved quantities
        lu = np.zeros(( 5, self.J, self.J ))
        lu[:, 3:-3, 3:-3] = -( f_s[:,3:-3,1:] - f_s[:,3:-3,:-1] )/self.dx - ( g_s[:,1:,3:-3] - g_s[:,:-1,3:-3] )/self.dy

        # forcing two ghost zones
        #lu[:, 2][:] = 0.0 ; lu[:,-2][:] = 0.0
        
        return lu

    def test_prob(self, problem, start, load_data=False):

        # sets up initial conditions
        # depending on test problem

        if problem == 'sod':

            # grid variables for sod tube
            self.ax = 0.0 ; self.bx = 1.0
            self.ay = 0.0 ; self.by = 1.0
            self.dx = (self.bx-self.ax)/float(self.J)
            self.dy = (self.by-self.ay)/float(self.J)

            # x array values
            self.x = self.ax + self.dx*(np.arange(self.J) + 0.5)
            self.y = self.ay + self.dy*(np.arange(self.J) + 0.5)

            self.gamma = 1.4

            # setting rho_L , P_L for x and y
            self.prim[0, :, :self.J/2] = 1.0 # density
            self.prim[1, :, :self.J/2] = 1.0 # pressure
            # setting rho_R , P_R for x and y
            self.prim[0, :, self.J/2:] = 0.1 # density
            self.prim[1, :, self.J/2:] = 0.125 # pressure

            self.prim[2, :, :] = 0.0 # v_x
            self.prim[3, :, :] = 0.0 # v_y

            #xx, yy = np.meshgrid(self.x, self.y)
            #plt.pcolormesh(xx, yy, self.prim[0])            
            #plt.show()
            #plt.close('all')

        if problem == 'blast':

            # grid variables for sod tube
            self.ax = -0.5 ; self.bx = 0.5
            self.ay = -0.5 ; self.by = 0.5
            self.dx = (self.bx-self.ax)/float(self.J)
            self.dy = (self.by-self.ay)/float(self.J)

            # x array values
            self.x = self.ax + self.dx*(np.arange(self.J) + 0.5)
            self.y = self.ay + self.dy*(np.arange(self.J) + 0.5)

            self.gamma = 5/3.

            # setting rho_L , P_L, v_L for x and y
            self.prim[0, :, :] = 1.0 # density
            self.prim[1, :, :] = 0.1 # pressure
            self.prim[2, :, :] = 0.0 # v_x
            self.prim[3, :, :] = 0.0 # v_y
            
                
            xx, yy = np.meshgrid(self.x, self.y)
            self.prim[1][ xx*xx + yy*yy < 0.01 ] = 10.0

            #plt.pcolormesh(xx, yy, self.prim[1])            
            #plt.show()
            #plt.close('all')

        if problem == 'kh':
            
            self.ax = 0.0 ; self.bx = 1.0
            self.ay = 0.0 ; self.by = 1.0

            self.dx = (self.bx-self.ax)/float(self.J)
            self.dy = (self.by-self.ay)/float(self.J)

            self.x = self.ax + self.dx*(np.arange(self.J)+0.5)
            self.y = self.ay + self.dy*(np.arange(self.J)+0.5)

            self.gamma = 5/3.
            w0 = 0.1 ; sigma = 0.05/np.sqrt(2)
            
            xx, yy = np.meshgrid(self.x, self.y)

            gx = w0*np.sin(4*np.pi*xx)
            fy = np.exp(-((yy-0.25)**2.)/(2*sigma*sigma) ) + np.exp(-((yy-0.75)**2.)/(2*sigma*sigma) )

            self.prim[3] = gx*fy

            # initial conditions
            # top
            #bottom
            self.prim[0] = 1.0
            self.prim[2] = -0.5

            self.prim[0][ (yy>0.25)*(yy<0.75) ] = 2.0
            self.prim[4][ (yy>0.25)*(yy<0.75) ] = 1.0
            self.prim[2][ (yy>0.25)*(yy<0.75) ] = 0.5
            
            # everywhere
            self.prim[1] = 2.5            

        if problem == 'pulse':
            
            # grid values for isentropic pulse
            self.a = 0.0 ; self.b = 2.0
            self.dx = (self.b-self.a)/float(self.J)
            # x array values
            self.x = self.a + self.dx*(np.arange(self.J) + 0.5)

            # constants for creating initial conditions
            rho_0 = 1.0 ; p_0 = 0.6 ; self.gamma = 5/3.
            alpha = 0.2 ; x_0 = 0.5 ; sigma = 0.4

            f = ( 1 - ((self.x - x_0)/sigma)**2. )**2
            f[ abs(self.x-x_0) > sigma ] = 0.0

            self.prim[0] = rho_0*(1 + alpha*f)
            self.prim[1] = p_0*(self.prim[0]/rho_0)**self.gamma
            cs_0 = np.sqrt( self.gamma*p_0/rho_0 )
            cs = np.sqrt( self.gamma*self.prim[1]/self.prim[0] )

            self.prim[2] = (2/(self.gamma-1))*( cs - cs_0 )

            self.s = np.log( (self.prim[1]/p_0)*(self.prim[0]/rho_0)**(-self.gamma))/(self.gamma-1)
            self.Lt = []


        if load_data:
            self.prim = pickle.load( open('/Users/austin/classes/comp_phys/finalproject-atm426/data/'+problem+'_data_'+str(start).zfill(4)+'.pkl', 'rb'))
            self.n = int(start)
        # initial values for conserved quantities
        self.u[0] = np.copy(self.prim[0]) # rho
        self.u[1] = self.prim[0]*self.prim[2] # rho * v_x
        self.u[2] = self.prim[0]*self.prim[3] # rho * v_y
        self.u[3] = self.prim[1]/(self.gamma - 1) + 0.5*self.prim[0]*(self.prim[2]*self.prim[2]+self.prim[3]*self.prim[3]) # E
        self.u[4] = self.prim[0]*self.prim[4]
        # initial values for x-dir fluxes
        self.f[0] = np.copy(self.u[1]) # rho * v_x
        self.f[1] = self.u[1] * self.prim[2] + self.prim[1] # rho*v_x^2 + P
        self.f[2] = self.u[1]*self.prim[3] # rho * v_x * v_y
        self.f[3] = self.prim[2]*( self.u[3] + self.prim[1] )
        self.f[4] = self.prim[0]*self.prim[4]*self.prim[2]

        # initial values for y-dir fluxes
        self.g[0] = np.copy(self.u[2]) # rho * v_y
        self.g[1] = np.copy(self.f[2]) # rho * v_x * v_y
        self.g[2] = self.u[2]*self.prim[3]+self.prim[1] # rho * v_y^2 + P
        self.g[3] = self.prim[3]*( self.u[3] + self.prim[1] )
        self.g[4] = self.prim[0]*self.prim[4]*self.prim[3]

        self.cs = np.sqrt( self.gamma * self.prim[1] / self.prim[0] )

    def plot_hydro(self, tf, problem, filename=None):

        # plots primitive variables

        if problem == 'blast':

            xx, yy = np.meshgrid(self.x, self.y)

            if (self.t0 < 0) and (self.t != 0.0):
                self.t0 = self.t                
                        
            r = ((self.t/self.t0)**(2/5.))*0.1

            #print self.t0
            circ1 = Circle((0,0), r, facecolor='None', edgecolor='k', lw=5, zorder=10)
            
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            plt.pcolormesh(xx[3:-3, 3:-3], yy[3:-3, 3:-3], self.prim[1][3:-3, 3:-3], vmin=0.8, vmax=3.0, cmap='jet')
            plt.colorbar()
            ax.add_patch(circ1)
            plt.tight_layout()
            plt.axis('tight')
            #plt.show()
            #plt.close('all')

        og_filename = filename
        if problem == 'kh':
            filename = og_filename

            xx, yy = np.meshgrid(self.x, self.y)
            plt.pcolormesh(xx[3:-3, 3:-3], yy[3:-3, 3:-3], self.prim[0][3:-3, 3:-3], vmin=1.0, vmax=2.1)
            
            plt.axis('tight')
            plt.axis('off')

            if filename is not None:
                plt.savefig('/Users/austin/classes/comp_phys/finalproject-atm426/imgs/extra/density_'+filename+str(self.n).zfill(4)+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
                print 'saved step '+str(self.n)

            plt.close('all')

            # plot passive scalar
            #cmap = colors.ListedColormap(['black', 'grey'])
            #bounds = [0.0, 1.0]
            #norm = colors.BoundaryNorm(bounds, cmap.N)
            #plt.pcolormesh(xx[3:-3, 3:-3], yy[3:-3, 3:-3], self.prim[4][3:-3, 3:-3], vmin=0.0, vmax=1.0, cmap=cmap)
            #plt.colorbar(ticks=[0,1])
            #plt.axis('tight')
            #plt.axis('off')

            #if filename is not None:
            #    plt.savefig('/Users/austin/classes/comp_phys/finalproject-atm426/imgs/extra/q_'+filename+str(self.n).zfill(4)+'.png', bbox_inches='tight', pad_inches=0, transparent=True)

            #print str(round(self.t, 4))+' / '+str(round(tf, 3))
            filename=None # dont save again

        if problem == 'sod':
            xx, yy = np.meshgrid(self.x, self.y)
            plt.pcolormesh(xx[3:-3, 3:-3], yy[3:-3, 3:-3], self.prim[0, 3:-3, 3:-3], cmap='jet', vmin=0, vmax=1)
            #plt.colorbar()
            plt.tight_layout()
            plt.axis('tight')
            
            #plt.show()
            #plt.close('all')

        if filename is not None:
            
            plt.savefig('/Users/austin/classes/comp_phys/finalproject-atm426/imgs/'+filename+str(self.n).zfill(4)+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
            print 'saved step '+str(self.n)


        print str(round(self.t, 4))+' / '+str(round(tf, 3))
            
        #else:
            #plt.show()
        plt.close('all')


    def error_plot(self, problem, error, savefig=False):
    
     from eulerExact import main

     N_arr = []                                                             
     L_rho = []                                                             
     L_P = []                                                               
     L_v = []
     s_arr = []

     for i in xrange(10):                                                   
            N = 100*(i+1.0)                                                            
            print N                                                        
                                                                               
            h = hydro(N)                                                 
            h.main_fn(problem, 0.1, filename=None)                               
            X, rho, v, P = main(problem, N, h.t-h.dt)                               
                 
            if savefig:
                plt.subplot(3, 1, 1)                                              
                plt.ylabel('density')                                             
                plt.plot(h.x, h.prim[0,:], 'b.', label='t='+str(round(h.t,3)))    
                plt.plot(X, rho, 'k--', label='exact solution')
                if problem == 'hsod':
                    plt.axis([0, h.x[-1], -0.01, 12.0])
                if problem == 'sod':
                    plt.axis([0, h.x[-1], -0.01, 1.2])
                if problem == 'pulse':
                    plt.axis([0, h.x[-1], 1.0, 1.3])                                
                plt.legend(numpoints=1, frameon=False)                            
                plt.locator_params(nbins=4)    

                plt.subplot(3, 1, 2)                                              
                plt.ylabel('pressure')                                            
                plt.plot(h.x, h.prim[1,:], 'y.')                                  
                plt.plot(X, P, 'k--')                                             
                if problem == 'hsod':
                    plt.axis([0, h.x[-1], -0.01, 110.0])
                if problem == 'sod':
                    plt.axis([0, h.x[-1], -0.01, 1.2])
                if problem == 'pulse':
                    plt.axis([0, h.x[-1], 0.6, 1.])
                plt.locator_params(nbins=4)                                       
                                                                               
                plt.subplot(3, 1, 3)                                              
                plt.ylabel('velocity')                                            
                plt.plot(h.x, h.prim[2,:], 'g.')                                  
                plt.plot(X, v, 'k--')                                             
                if problem == 'hsod':
                    plt.axis([0, h.x[-1], -0.01, 10.0])
                if problem == 'sod':
                    plt.axis([0, h.x[-1], -0.01, 1.2])
                if problem == 'pulse':
                    plt.axis([0, h.x[-1], 0.0, 0.2])
                plt.locator_params(nbins=4)                                            
                plt.tight_layout()

                plt.savefig('/Users/austin/classes/comp_phys/finalproject-atm426/ho_'+problem+'_convergence_N'+str(int(N))+'.eps', bbox_inches='tight')
                plt.close('all')          

                                         
            N_arr.append(N)
            if problem=='pulse':
                s_arr.append(h.Lt[-1])

            # L2 error
            if error == 'l2':
                L_rho.append( h.dx*np.sqrt(np.sum( ( h.prim[0] - rho )**2.)))              
                L_P.append( h.dx*np.sqrt(np.sum( ( h.prim[1] - P )**2. )))                  
                L_v.append( h.dx*np.sqrt(np.sum( ( h.prim[2] - v )**2.)))                  
            # L1 error 
            if error == 'l1':
                L_rho.append( h.dx*np.sum( abs( h.prim[0] - rho )))              
                L_P.append( h.dx*(np.sum( abs( h.prim[1] - P ))))                  
                L_v.append( h.dx*(np.sum( abs( h.prim[2] - v ) )))                  
                                     

     N_arr = np.array(N_arr)                                                
     L_rho = np.array(L_rho)                                                
     L_P = np.array(L_P)                                                    
     L_v = np.array(L_v)         
     s_arr = np.array(s_arr)
     
                                                                               
     N_x = np.linspace(np.log10(N_arr).min(), np.log10(N_arr).max(), 100)   
     rho_coeffs = np.polyfit(np.log10(N_arr), np.log10(L_rho), 1)           
     y_rho = rho_coeffs[0]*N_x + rho_coeffs[1]                              
                                                                               
     P_coeffs = np.polyfit(np.log10(N_arr), np.log10(L_P), 1)               
     y_P = P_coeffs[0]*N_x + P_coeffs[1]                                    
                                                                               
     v_coeffs = np.polyfit(np.log10(N_arr), np.log10(L_v), 1)               
     y_v = v_coeffs[0]*N_x + v_coeffs[1]                                    
     
     if problem == 'pulse':
         s_coeffs = np.polyfit(np.log10(N_arr), np.log10(s_arr), 1)
         y_s = s_coeffs[0]*N_x + s_coeffs[1]
                                                   
     plt.subplot(3, 1, 1)                                                   
     plt.ylabel('density error') ; plt.xlabel('grid size')                  
     plt.plot(np.log10(N_arr), np.log10(L_rho), 'b+', markersize=14)        
     plt.plot(N_x, y_rho, 'k--', label='m='+str(round(rho_coeffs[0],3)))    
     plt.legend(numpoints=1, frameon=False)                                 
     plt.locator_params(nbins=4)                                            
                                                                               
     plt.subplot(3, 1, 2)                                                   
     plt.ylabel('pressure error') ;  plt.xlabel('grid size')                
     plt.plot(np.log10(N_arr), np.log10(L_P), 'y+', markersize=14)          
     plt.plot(N_x, y_P, 'k--', label='m='+str(round(P_coeffs[0],3)))        
     plt.legend(numpoints=1, frameon=False)                                 
     plt.locator_params(nbins=4)                                            
                                                                               
     plt.subplot(3, 1, 3)                                                   
     plt.ylabel('velocity error') ;  plt.xlabel('grid size')                
     plt.plot(np.log10(N_arr), np.log10(L_v), 'g+', markersize=14)          
     plt.plot(N_x, y_v, 'k--', label='m='+str(round(v_coeffs[0],3)))        
     plt.legend(numpoints=1, frameon=False)                                 
     plt.locator_params(nbins=4)            

     plt.tight_layout()
     
     plt.show()
     plt.close('all')

     if problem=='pulse':
         plt.ylabel('specific entropy') ; plt.xlabel('grid size')
         plt.plot(np.log10(N_arr), np.log10(s_arr), 'r+', markersize=14)
         plt.plot(N_x, y_s, 'k--', label='m='+str(round(s_coeffs[0],3)))        
         plt.legend(numpoints=1, frameon=False)                                 
         plt.locator_params(nbins=4)
         plt.show()
         plt.close('all')

     return N_arr, L_rho, L_P, L_v
