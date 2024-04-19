import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from queue import Queue
import os
class PlonkaDenoiseMasterClass:
    """ Class for Plonka-Hoch Denoising Method(s) Proposed in the paper- "CONVERGENCE OF AN ITERATIVE NONLINEAR SCHEME FOR DENOISING OF PIECEWISE CONSTANT IMAGES".
    
    The class SCPractical contains the following methods:

    1. pl_iter_method:  Implements the Plonka-Hoch Denoising Method.
    2. iterate: Implements the iteration function Naively. 'flag = False'
    3. iterate_fast: implements the iteration function in a faster way.'flag = True'
    4. block_finding: Finds the blocks in the image. It uses the method find_blocks_func.
    5. mean_perBlock: Calculates the mean  of each block.
    6. mean_filter: Implements the mean filter as proposed by Plonka Hoch.
    7. median_filter: Implements the median filter as proposed by Plonka Hoch.
    8. get_neighbors: Calculates the neighbors of a pixel in the image. It uses the method get_neighbors_func which is a helper function for get_neighbors 
        where we can pass the threshold_flag to get the neighbors of a pixel based on the threshold value
       
    Parameters: image_path, resize_shape, sigma, theta, alpha, num_iter, flag, boundary_condition.
      
    Attributes: im, arr, resized_image, H, W, sigma, theta, alpha, num_iter, noisy_image.
        
    Methods: pl_iter_method, iterate, block_finding, mean_perBlock, mean_filter, median_filter, get_neighbors. 
         
         """
    def __init__(self, image_path, resize_shape=(50, 50), sigma=10, theta=15, alpha=0.1, num_iter=10, flag=False):
        self.im = Image.open(image_path).convert('L')
        self.arr = np.array(self.im)
        self.resized_image = cv2.resize(self.arr, resize_shape)
        self.H, self.W = resize_shape
        self.sigma = sigma
        self.theta = theta
        self.alpha = alpha
        self.num_iter = num_iter
        self.flag = flag



    #Step[1] = plonka_hoch iteration method; flag = True for faster implementation and flag = False for naive implementation
    def pl_iter_method(self, sigma=None, theta=None, alpha=None, num_iter=None, flag=None):
        '''Variable Description:
        1.sigma = noise standard deviation
        2.theta = threshold value
        3.alpha = alpha value
        4.num_iter = number of iterations
        5.flag = True for faster implementation and flag = False for naive implementation
         '''
        if sigma is None:
            sigma = self.sigma
        if theta is None:
            theta = self.theta
        if alpha is None:
            alpha = self.alpha
        if num_iter is None:
            num_iter = self.num_iter
        if flag is None:
            flag = self.flag

        # adding noise to the image
        np.random.seed(42)
        noise = np.random.normal(0, sigma, size=(self.W, self.H))
        self.noise_alter = noise
        self.noise_called =self.resized_image + noise
        noisy_im = self.resized_image + noise
        self.noisy_image = noisy_im
        iter_img = noisy_im
        # st_dev = np.std(self.resized_image)
        # noise = np.random.normal(0, st_dev, size=(self.W, self.H))
        # self.noise_alter = noise
        # self.noise_called =self.resized_image + noise
        # noisy_im = self.resized_image + noise
        # self.noisy_image = noisy_im
        # iter_img = noisy_im
        # print(noise)

        # iteration loop
        for k in range(num_iter):
           
            if(flag):
                iter_img = self.iterate_fast(iter_img, theta, alpha)
                # print('Periodic')
            else:
                iter_img = self.iterate(iter_img, theta, alpha)

        
        print(f'Iteration {k+1} completed.')

        return iter_img, self.noise_called
    
    #iteration function Naive approach
    def iterate(self, iter_img, theta, alpha):
        old_img = iter_img.copy()
        for i in range(self.H):
            for j in range(self.W):

                for r in range(-1, 2):
                    for s in range(-1, 2):
                        if (r, s) != (0, 0):
                            i_p = (i + r) % self.H #periodic boundary condition
                            j_p = (j + s) % self.W
                            iter_img[i, j] += alpha * self.T_theta(old_img[i_p, j_p] - old_img[i, j], theta) / (r**2 + s**2)
        return iter_img


    #Faster implementation of the iteration function
    def iterate_fast(self, iter_img, theta, alpha):
        # H, W = iter_img.shape
        iter_img_new = iter_img.copy()

        directions = [(r, s) for r in (-1, 0, 1) for s in (-1, 0, 1) if (r, s) != (0, 0)]
        weights = [1/(r**2 + s**2) for r, s in directions]

        for (r, s), weight in zip(directions, weights):
            shifted_img = np.roll(iter_img, shift=(r, s), axis=(0, 1))
            iter_img_new += alpha * weight * self.T_theta(shifted_img - iter_img, theta)

        return iter_img_new
    


    @staticmethod
    def T_theta(x, theta):
        return np.where(np.abs(x) < theta, x, 0)
    
    ############################################### Step[1] end ##############################################
  

    #Step[2] = block finding + mean filter  as proposed by Plonka Hoch

    #block finding
    def find_blocks(self, img, theta=None):
        '''Variable Description:
         1.img =iterated image
          2.theta = threshold value '''
        
        if theta is None:
            theta = self.theta
        return self.find_blocks_func(img, theta)
    
    #block finding function
    def find_blocks_func(self, img, theta):
        rows, cols = img.shape
        blocks = []
        visited = set()
        # print(theta)
        # print(img.shape)

        for x in range(rows):
            for y in range(cols):
                if (x, y) not in visited:
                    block = []
                    queue = Queue()
                    queue.put((x, y))

                    while not queue.empty():
                        current_pixel = queue.get()
                        if current_pixel not in visited:
                            visited.add(current_pixel)
                            block.append(current_pixel)
                            neighbors = self.get_neighbors(*current_pixel, img, theta)
                            for neighbor in neighbors:
                                queue.put(neighbor)

                    blocks.append(block)

        return blocks
    
    

    #determine the mean of each block
    def mean_perBlock(self, B, img):
        ''' Varibale Description:
            1. B: list of blocks
            2. img: iterated image'''
        
        means = []
        for block in B:
            block_values = [img[x, y] for x, y in block]
            mean = np.mean(block_values)
            # median = np.median(block_values)
            means.append((mean))
        return means
    
    #mean filter function as proposed by Plonka Hoch
    def mean_filter(self, img, means, B):
        ''' Varibale Description:
            1. img: iterated image
            2. means: list of means of each block
            3. B: list of blocks '''
        
        
        new_img = img.copy()
        for i, block in enumerate(B):
            mean = means[i]
            for _, coord in enumerate(block):
                x, y = coord
                new_img[x, y] = mean
        return new_img
    
   ############################################## Step[2] end ##############################################

    #Step[3] = median filter

    #median filter as proposed by Plonka Hoch
    def median_filter(self, img, blocks, theta=None):
        ''' Varibale Description:
            1. img: iterated image
            2. blocks: list of blocks
            3. theta: threshold value '''
        
        if theta is None:
            theta = self.theta
        return self.median_filter_func(img, blocks, theta)
    
    #median filter function
    def median_filter_func(self, img, blocks, theta):
        filtered_img = img.copy()
        for block in blocks:
            if len(block) < 6:
                for x, y in block:
                    neighbors = self.get_neighbors(x, y, img, theta, threshold_flag=False)
                    neighbor_values = [img[nx, ny] for nx, ny in neighbors]
                    median = np.median(neighbor_values)
                    filtered_img[x, y] = median
        return filtered_img
    
    ############################################### Step[3] end ##############################################


    
    ######################### Very Essential Function which calculate the Neighbors of a pixel ###############################

    #get neighbors of a pixel in the image
    def get_neighbors(self, x, y, img, theta=None, threshold_flag=True):
        ''' Varibale Description:
            1. x: x coordinate of the pixel
            2. y: y coordinate of the pixel
            3. img: iterated image
            4. theta: threshold value
            5. threshold_flag: flag to determine whether to use threshold or not'''
        
        if theta is None:
            theta = self.theta
        return self.get_neighbors_func(x, y, img, theta, threshold_flag)
    
    #get neighbors main function
    def get_neighbors_func(self,x, y, img, theta, threshold_flag=True):
        neighbors = []
        rows, cols = img.shape

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                new_x, new_y = (x + i) % rows, (y + j) % cols  # Use modulus operator to add periodicity
                if threshold_flag:
                    if abs(img[x, y] - img[new_x, new_y]) < theta:
                        neighbors.append((new_x, new_y))
                else:
                    neighbors.append((new_x, new_y))
            # print("periodic")
 

        return neighbors
    ############################################### Extra- Fine Tuning Parameter ##############################################






class PlonkaHochDenoising(PlonkaDenoiseMasterClass):
    '''
    This class is for the practical part of the project. It inherits the `PlonkaMasterClass` class. It has the following method: 

    1. PNSR:  PNSR between the original and reconstructed image.
    2. compute_contrast_with_neighbors: computes the contrast of each pixel with its neighbors 
    3. plonka_method: It calls all the methods in the PlonkaHoch_Denoising class and plots the results. 

    Inherited: 
    Parameters: image_path, resize_shape, sigma, theta, alpha, num_iter, flag, boundary_condition.
      
    Attributes: im, arr, resized_image, H, W, sigma, theta, alpha, num_iter, noisy_image.
        
    Methods: pl_iter_method, iterate, block_finding, mean_perBlock, mean_filter, median_filter, get_neighbors.

    Descriptions: 
    1. pl_iter_method:  Implements the Plonka-Hoch Denoising Method.
    2. iterate: Implements the iteration function Naively. 'flag = False'
    3. iterate_fast: implements the iteration function in a faster way.'flag = True'
    4. block_finding: Finds the blocks in the image. It uses the method find_blocks_func.
    5. mean_perBlock: Calculates the mean  of each block.
    6. mean_filter: Implements the mean filter as proposed by Plonka Hoch.
    7. median_filter: Implements the median filter as proposed by Plonka Hoch.
    8. get_neighbors: Calculates the neighbors of a pixel in the image. It uses the method get_neighbors_func which is a helper function for get_neighbors 
        where we can pass the threshold_flag to get the neighbors of a pixel based on the threshold value 
    
    '''


    # def PNSR(self, original, reconstructed):
    #     mse = np.mean((original - reconstructed)**2)
    #     return 10 * np.log10(1 / mse)
    def SNR(self, original):
            ''' Compute the SNR of denioised image.
             Varibale Description:
            1. original:  image
            2. noise_alter: noise created by the algorithm

            Returns:
            20 * np.log10(signal_error / noise) as described in the paper
            '''
        # signal = np.sum(original**2)
        # noise = np.sum((original - reconstructed)**2)
        # return 10 * np.log10(signal / noise)
            signal_error = np.linalg.norm(original - np.mean(original)) 
            noise = np.linalg.norm(self.noise_alter) # noisy created   
            return 20 * np.log10(signal_error / noise)

    def compute_contrast_with_neighbors(self,img, theta):
        ''' Compute the contrast of each pixel with its neighbors.

        Args:
        img (ndarray): The input image.
        theta (float): The threshold parameter.

        Returns:
        c (float): The minimum non-zero difference representing the contrast.
    '''
        differences = []
        rows, cols = img.shape
        for x in range(rows):
            for y in range(cols):
                neighbors = self.get_neighbors(x, y, img, theta, threshold_flag=False)
                for nx, ny in neighbors:
                    difference = abs(float(img[x, y]) - float(img[nx, ny]))
                    differences.append(difference)

        # Compute the minimum non-zero difference
        differences = np.array(differences)
        c = np.min(differences[differences > 0])
        return c  
    

    def plonka_method(self, shrinkage_param):
        ''' This method calls all the necessary methods in the SCPractical class and plots the results. 
        Inherited from class:
        -Parameters:  
            1.``sigma``:(Float) Noise Your are adding(defined on instantiating the class)
            2.``theta``:(Float) Shrinkage parameter
            3.``alpha``:(Float) Smoothing Parameter 
            4.``num_iter``:(Integer) Interation you want to do 
            5.``flag`` :(Boolean) Faster calculation or naive calculation

        New Parameter for method:
            1.``shrinkage_param``:(Float) Finding blocks primarily, which is later important for mean and median value procedure.
        
        Plots:
            1.``img_f_median``:(ndarray) Denoised image
            2.``snr``:(Float) Signal to Noise Ratio
            3.``c``:(Float) Contrast of the image
            4.``iteration``:(ndarray) Image after iteration
            5.``img_f_mean``:(ndarray) Image after mean filter
            6.``resized_image``:(ndarray) Resized Original image
            6.``noisy_image``:(ndarray) Noisy image

        '''
        
        # method based on Plonka Hoch - iteration
        iteration,_ = self.pl_iter_method(self.sigma, self.theta, self.alpha, self.num_iter, self.flag)


        # find blocks on new image after iteration
        blocks = self.find_blocks(iteration, theta= shrinkage_param)
        # print('Number of blocks: ', len(blocks))

        # mean & median of the each block
        means_medians = self.mean_perBlock(blocks, iteration)

        # mean filter
        img_f_mean = self.mean_filter(iteration, means_medians, blocks)

        # median filter
        img_f_median = self.median_filter(img_f_mean, blocks, theta= shrinkage_param)

        snr = self.SNR(img_f_median)
        # img= self.im

        c = self.compute_contrast_with_neighbors(self.arr, self.theta)
        # print('Contrast: ', c,'\n', c/self.sigma)
        # print(1,np.max(self.arr))
        # print(2,np.max(self.resized_image))

        # plotting results for each steps
        plt.rcParams['axes.titlesize'] = 15
        fig, ax = plt.subplots(1, 5, figsize=(15, 15))
        ax[0].imshow(self.resized_image, cmap='gray')
        ax[0].set_title('Original Image')

        ax[1].imshow(self.noisy_image, cmap='gray')
        ax[1].set_title('Noise added Image')

        ax[2].imshow(iteration, cmap='gray')
        ax[2].set_title(f'After Iteration- {self.num_iter}')

        ax[3].imshow(img_f_mean, cmap='gray')
        ax[3].set_title('Mean Filtered Image')

        ax[4].imshow(img_f_median, cmap='gray')
        ax[4].set_title(f'Median Filtered Image, SNR:{snr:.2f}dB')

        title_text = 'Image Analysis Results: \n' + 'contrast/noise =' +  str(c/np.std(self.noise_alter)) 
        fig.suptitle(title_text, y=0.78)

        # plt.title(title_text) 

        for i in ax.ravel():
            i.axis('off')
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    img_path = os.path.dirname(os.path.abspath(__file__))
    
    resize_shape = (200,200) #change the shape of the image
    sigma =10
    theta = 40 #if c/sigma<1.5 then theta = np.inf
    alpha = 0.15
    num_iter =3
    flag = True #for fast computation- True or False
    
    instance = PlonkaHochDenoising(
        img_path + '\\example-1.png',
        resize_shape=resize_shape,
        sigma= sigma,
        theta= theta,
        alpha= alpha,
        num_iter= num_iter,
        flag= flag,

    )
    instance.plonka_method(shrinkage_param= 10)

    

