from sympy import Matrix, Rational
import matplotlib.pyplot as plt
import numpy as np
import sys

# TERMINOLOGY:
# - nodes: always the fixed points upon which values are set
# - points: variable coordinates typically used to interpolate on


#nodes are rationals,Nodes are float64 use them for calc !!!!!!!!!!!!!!!!!


#we can init this class in two ways with pre=True or true=False,this is going to have an inpact on the evaluation methods:
#-->False:works like usual 
#-->True:you need to init with all the points where you want your interpolation that you know a priori.when you perform interpoaltion
#   you need to put points=None always you can't use the method with other points(you dont' really have to)

print('interpolator_lib imported')



# TODO introduce also second derivative 
# clear structure so you can have up to order 2 
#
#

class interpolator:
    def __init__(self, r: int,verbose:bool,pre:bool,points):
        """
        input=degree,verbose (True or False to shoe the poly)
        """
        self.r = r

        #to perform loops easily 
        self.powers= self.generate_powers(r)
        self.powers_dx,self.powers_dy=self.generate_powers_der(1)
        self.powers_d2x,self.powers_d2y=self.generate_powers_der(2)

        print(self.powers)

        print()
        print(self.powers_dx)
        print(self.powers_dy)

        print()
        print(self.powers_d2x)
        print(self.powers_d2y)


        #generate second powers

        self.nodes,self.Nodes,self.n = self.generate_interp_nodes(r)
        
        if bool:
            self.plot_nodes()


        #matrices that rapresent the polynomials
        self.M ,self.M_dx,self.M_dy,self.M_d2x,self.M_d2y= self.generate_matrices(verbose)

        self.pre=pre

        #to check if you want or not pre_eval and if you have provided the correct arg in the init 
        if(pre==False):
            pass
        else:
            if(points is None):
                    raise ValueError("'you didnt provide points(np.array) for the pre_eval!. The script will terminate.")
            else:
                self.eval_powers_a_priori(points)
            




    def plot_nodes(self) -> None:
        """
        plot the nodes used for the basis on the ref triangle
        """
        print("degree = ",self.r," , local dof = ",self.n)

        print(self.nodes)

        plt.figure()
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1])
        plt.grid()
        plt.show()

    def generate_powers(self, r: int) -> list:
        """
        r: int = degree
        generates the powers of the polynomial in x and y up to order r 
        """
        return [(j, i) for i in range(0, r + 1) for j in range(0, r + 1) if i + j <= r]
    
    def generate_powers_der(self,I:int):
        """
        generate  the correct powers for the derivatives and store it 
        """
        der_x=[]
        der_y=[]

        for (i,j) in self.powers:
            if (i-I)>=0:
                der_x.append(((i-I),j))
            if (j-I)>=0:
                der_y.append((i,(j-I)))

        return der_x,der_y

    def generate_interp_nodes(self, p):
        """
        generates the nodes used for the basis functions 
        """
        n = (p + 1) * (p + 2) // 2
        nodes = [
            [Rational(j, p), Rational(i, p)]
            for i in range(0, p + 1)
            for j in range(0, p + 1)
            if i + j <= p
        ]
        return np.array(nodes),np.array(nodes,dtype=np.float64),n
    


    def print_polynomial(self,sol,pairs,index) -> None:
        """singol polynomial print"""

        s='basis function number : '
        print("\033[1m" + s + "\033[0m",index+1)
        print()

        for ii,(i,j) in enumerate(pairs):
            s1="\033[1m" + "x^"+str(i) + "\033[0m"
            s2="\033[1m" + "y^"+str(j) + "\033[0m"
            print('\033[91m'+'\033[4m'+str(sol[ii])+'\033[0m'+'\033[0m',s1,s2,end=' ')
        print()
        print()
        



    def generate_coefficients(self, evaluation_matrix, verbose: bool):
        coeff = []


        if verbose :
            print('basis:')
        for i in range(self.n):
            b = [Rational(0) for j in range(self.n)]
            b[i] = Rational(1)
            b = Matrix(b)

            res = evaluation_matrix.solve(b)
            coeff.append(res)

            if verbose:
                self.print_polynomial(res,self.powers,i)

        return coeff


    #change this last two
    def generate_matrices(self,verbose:bool):

        list=[]
        for ii in range(self.n):
            temp=[]
            for jj,(i,j) in enumerate(self.powers):
            # A[ii,jj]=((points[ii][0])**i )*((points[ii][1])**j)
                temp.append(((self.nodes[ii][0])**i )*((self.nodes[ii][1])**j))
            list.append(temp)

        A = Matrix(list)


        coeffs = self.generate_coefficients(A,verbose)
        coeffs_dx,coeffs_dy=self.generate_M_der(coeffs,verbose,1)
        coeffs_d2x,coeffs_d2y=self.generate_M_der(coeffs,verbose,2)

        coeffs=np.reshape(np.array(coeffs, dtype=np.float64), (self.n, self.n))

        coeffs_dx=np.reshape(np.array(coeffs_dx, dtype=np.float64),(self.n, len(self.powers_dy)))
        coeffs_dy=np.reshape(np.array(coeffs_dy, dtype=np.float64), (self.n, len(self.powers_dy)))

        coeffs_d2x=np.reshape(np.array(coeffs_d2x, dtype=np.float64),(self.n, len(self.powers_d2x)))
        coeffs_d2y=np.reshape(np.array(coeffs_d2y, dtype=np.float64), (self.n, len(self.powers_d2y)))        

        return coeffs,coeffs_dx,coeffs_dy,coeffs_d2x,coeffs_d2y
    
    
    
    def generate_M_der(self, M: Matrix,verbose:bool,I:int):
        coeff_dx=[]
        coeff_dy=[]
        if verbose:
            print('basis :')
        for ii in range(self.n):
            temp_x=[]
            temp_y=[]
            for jj,(i,j) in enumerate(self.powers):
                if (i-I)>=0:
                    if(I==1):
                        temp_x.append(Rational(i,1)*M[ii][jj])
                    else:
                        temp_x.append(Rational(i,1)*Rational(i-1,1)*M[ii][jj])
                if (j-I)>=0:
                    if(I==1):
                        temp_y.append(Rational(j,1)*M[ii][jj])
                    else:
                        temp_y.append(Rational(j,1)*Rational(j-1,1)*M[ii][jj])

            if verbose:
                if(I==1):
                    print("d"+str(I)+"x")
                    self.print_polynomial(temp_x,self.powers_dx,ii)
                    print("d"+str(I)+"y")
                    self.print_polynomial(temp_y,self.powers_dy,ii)
                else:
                    print("d"+str(I)+"x")
                    self.print_polynomial(temp_x,self.powers_d2x,ii)
                    print("d"+str(I)+"y")
                    self.print_polynomial(temp_y,self.powers_d2y,ii)
                print()
                

            coeff_dx.append(temp_x)
            coeff_dy.append(temp_y)

        return coeff_dx,coeff_dy
    




    def eval(self,M,points,val,pairs):
        """
        generic function that interpolates a set of 2d nodes that are fixed in generic points inside the ref triangle,
        -M is the matrix with coeff of all the basis function in each row 
        -points is a np.array of size (n_points,2)
        -val is a np array of size (n_nodes,1)

        the output will be an col vecotr of size (n_points,1)
          
        """
 
        x=np.ones((np.shape(points)[0],len(pairs)),dtype=np.float64)

        for ii,(i,j) in enumerate(pairs):
            x[:,ii]=(points[:,0]**i)*(points[:,1]**j)
        

        return x @ M.T @val


    
    def eval_pre(self,B,val):
        return B @ val

        




    #use this method if you want the eval of points that you don't know a priori 
    def interpolate(self,points,val):
        """ points where you want to intepolate,val are the values at the fixed nodes 
            if you set pre=True pass and empty second arguments as points """
 
        return self.eval(self.M,points,val,self.powers)
    
    def interpolate_dx(self,points,val):
        return self.eval(self.M_dx,points,val,self.powers_dx)
    
    def interpolate_dy(self,points,val):
        return self.eval(self.M_dy,points,val,self.powers_dy)
    
    def interpolate_d2x(self,points,val):
        return self.eval(self.M_d2x,points,val,self.powers_d2x)
    
    def interpolate_d2y(self,points,val):
        return self.eval(self.M_d2y,points,val,self.powers_d2y)
    






    #use this methods if you know a priori the points of the eval 
    def interpolate_pre(self,val):
        """ points where you want to intepolate,val are the values at the fixed nodes 
            if you set pre=True pass and empty second arguments as points """
 
        return self.eval_pre(self.Base,val)
    
    def interpolate_pre_dx(self,val):
        return self.eval_pre(self.Base_dx,val)
    
    def interpolate_pre_dy(self,val):
        return self.eval_pre(self.Base_dy,val)

    def interpolate_pre_d2x(self,val):
        return self.eval_pre(self.Base_d2x,val)
    
    def interpolate_pre_d2y(self,val):
        return self.eval_pre(self.Base_d2y,val)
    





    #just computed one 
    def eval_powers_a_priori(self,points)->None:

        self.x_pre=np.ones((np.shape(points)[0],len(self.powers)),dtype=np.float64)

        self.dx_pre=np.ones((np.shape(points)[0],len(self.powers_dx)),dtype=np.float64)
        self.dy_pre=np.ones((np.shape(points)[0],len(self.powers_dy)),dtype=np.float64)

        self.d2x_pre=np.ones((np.shape(points)[0],len(self.powers_d2x)),dtype=np.float64)
        self.d2y_pre=np.ones((np.shape(points)[0],len(self.powers_d2y)),dtype=np.float64)
        


        for ii,(i,j) in enumerate(self.powers):
            self.x_pre[:,ii]=(points[:,0]**i)*(points[:,1]**j)    
    

        for ii,(i,j) in enumerate(self.powers_dx):
            self.dx_pre[:,ii]=(points[:,0]**i)*(points[:,1]**j)  

        for ii,(i,j) in enumerate(self.powers_dy):
            self.dy_pre[:,ii]=(points[:,0]**i)*(points[:,1]**j)  


        for ii,(i,j) in enumerate(self.powers_d2x):
            self.d2x_pre[:,ii]=(points[:,0]**i)*(points[:,1]**j)  

        for ii,(i,j) in enumerate(self.powers_d2y):
            self.d2y_pre[:,ii]=(points[:,0]**i)*(points[:,1]**j)  



        self.Base=self.x_pre@self.M.T

     

        self.Base_dx=self.dx_pre@self.M_dx.T
        self.Base_dy=self.dy_pre@self.M_dy.T

        self.Base_d2x=self.d2x_pre@self.M_d2x.T
        self.Base_d2y=self.d2y_pre@self.M_d2y.T        





    def change_of_coordinates(self,vertices:list):
        """input: vertices where you want to traslate your points
            output:matrices that allow you to do the trasformation
            to perform the trasformation you need a numpy vector of size (2,npoints)
             --> x=B @ col vec of a point  +c you will get a col vec as output,
             the same goes for B_D, B_DD """
        B=np.zeros((2,2),dtype=np.float64)
        c=np.zeros((2,1),dtype=np.float64)



        B_D=np.zeros((2,2),dtype=np.float64)
        B_DD=np.zeros((2,2),dtype=np.float64)

        B[0][0]=(vertices[1, 0] - vertices[0, 0])
        B[0][1]=(vertices[2, 0] - vertices[0, 0])
        B[1][0]=(vertices[1, 1] - vertices[0, 1])
        B[1][1]=(vertices[2, 1] - vertices[0, 1])
        c[0]=vertices[0, 0]
        c[1]=vertices[0, 1]

        det = (vertices[1, 0] - vertices[0, 0]) * (vertices[2, 1] - vertices[0, 1]) - (
            vertices[2, 0] - vertices[0, 0]
        ) * (vertices[1, 1] - vertices[0, 1])


        #B^-T

        B_D[0][0]=(vertices[2, 1] - vertices[0, 1])
        B_D[0][1]=-(vertices[1, 1] - vertices[0, 1])
        B_D[1][0]=-(vertices[2, 0] - vertices[0, 0])
        B_D[1][1]=(vertices[1, 0] - vertices[0, 0])
       
        B_D=B_D/det

        #for the second derivative 
        B_DD=np.square(B_D)


        return B,c,det,B_D,B_DD
