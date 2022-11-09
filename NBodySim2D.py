import numpy as np
import matplotlib.pyplot as plt
import moderngl
import pygame
from matplotlib import cm
import time
import cv2


class Shower():
    """
    This object handles the display using PyGame. It recieves a simulator object and performs the stepping
    during the mainloop.
    Unfortunately, the arrays are sent to the CPU and then to pygame which is kinda stupid but oh well..
    """
    def __init__(self, simulator, magnification=1, make_video=False, num_vid_frames=0):
        """
        Gets a simulator object and a magnification factor (if you want one pixel to be more than one screen pixel).
        if one wants to record a video, just use make_video, video is made using opencv-python
        """
        # initializing shit
        self.simulator = simulator
        self.sim_size = self.simulator.show_size
        self.size = (self.sim_size[0] * magnification, self.sim_size[1] * magnification)
        self.gameDisplay = pygame.display.set_mode(self.size)
        pygame.display.set_caption('FLYING THINGS')
        self.make_video = make_video
        self.num_vid_frames = num_vid_frames

        # initializing stuff for the mouse control
        self.center = [0.5, 0.5]
        self.scale = 1.0
        self.mouse_pos = (0, 0)
        self.mouse_right_pressed = False
        self.mouse_left_pressed = False
        self.time_step_multiplier = 1.0

    def scroll(self, event):
        """
        handles zooming when mouse is scrolled
        """
        # zoom factor is 0.9 per scroll
        alpha = 0.9**event.y
        pos_x, pos_y = self.mouse_pos[0] * 2 / self.size[0] - 1, -self.mouse_pos[1] * 2 / self.size[0] + 1
        self.center[0] += (1 - alpha) * 0.5 * pos_x * self.scale
        self.center[1] += (1 - alpha) * 0.5 * pos_y * self.scale
        self.scale *= alpha
        self.scale = np.clip(self.scale, 0.000001, 1.0)
        # center needs to be recentered such that the view doesnt go over the edges
        self.center[0] = np.clip(self.center[0], 0.5 - 0.5 * (1 - self.scale), 0.5 + 0.5 * (1 - self.scale))
        self.center[1] = np.clip(self.center[1], 0.5 - 0.5 * (1 - self.scale), 0.5 + 0.5 * (1 - self.scale))

    def move_centers_along_pixels(self, shift):
        """
        shifts the view plane along x and y speicified by shift
        """
        shift_x, shift_y = shift[0] / self.size[0], -shift[1] / self.size[0]
        self.center[0] -= shift_x * self.scale
        self.center[1] -= shift_y * self.scale
        self.center[0] = np.clip(self.center[0], 0.5 - 0.5 * (1 - self.scale), 0.5 + 0.5 * (1 - self.scale))
        self.center[1] = np.clip(self.center[1], 0.5 - 0.5 * (1 - self.scale), 0.5 + 0.5 * (1 - self.scale))

    def mouse_down_event(self, event):
        """
        pycharm event shit
        """
        if event.button == 3:
            self.mouse_right_pressed = True
        if event.button == 1:
            self.mouse_left_pressed = True
            
    def mouse_up_event(self, event):
        """
        pycharm event shit
        """
        if event.button == 3:
            self.mouse_right_pressed = False
        if event.button == 1:
            self.mouse_left_pressed = False

    def mouse_motion_event(self, event):
        """
        pycharm event shit
        """
        self.mouse_pos = event.pos
        if self.mouse_right_pressed:
            self.move_centers_along_pixels(event.rel)
        if self.mouse_left_pressed:
            y_shift = event.rel[1]
            self.time_step_multiplier *= 0.99 ** y_shift

    def window_leave(self, event):
        """
        pycharm event shit
        """
        self.mouse_right_pressed = False
        self.mouse_left_pressed = False

    def mainloop(self):
        """
        starts the window and time stepping
        """
        if self.make_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            cv_video_writer = cv2.VideoWriter('NBodyVideo.mp4', fourcc, 30.0, (self.size[0], self.size[1]))


        crashed = False
        frame_counter = 0
        
        while not crashed:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down_event(event)
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_up_event(event)
                if event.type == pygame.MOUSEMOTION:
                    self.mouse_motion_event(event)
                if event.type == pygame.WINDOWLEAVE:
                    self.window_leave(event)
                if event.type == pygame.MOUSEWHEEL:
                    self.scroll(event)
                if event.type == pygame.QUIT:
                    crashed = True
                    break

            # bbox for view is computed
            bbox = (self.center[0] - 0.5 * self.scale, self.center[0] + 0.5 * self.scale,
                    self.center[1] - 0.5 * self.scale, self.center[1] + 0.5 * self.scale)

            # the time for each solve is printed
            t0 = time.perf_counter()
            show_array = self.simulator.step(bbox, self.time_step_multiplier)
            t1 = time.perf_counter()
            print(frame_counter, t1 - t0)

            if self.make_video and frame_counter < self.num_vid_frames:
                if frame_counter < self.num_vid_frames:
                    cv_video_writer.write(show_array.transpose(1, 0, 2)[:, :, ::-1])
                elif frame_counter == self.num_vid_frames:
                    cv_video_writer.release()

            # pycharm display stuff
            quantity_surf = pygame.transform.scale(pygame.pixelcopy.make_surface(show_array[:, ::-1]), self.size)
            self.gameDisplay.blit(quantity_surf, (0, 0))
            pygame.display.update()
            frame_counter += 1
            
        pygame.quit()


class NBodySimulator:
    """
    The main simulator object, it handles all the stuff with shaders and such
    """
    def __init__(self, num_bodys, mesh_exponent, g_constant, time_step, show_size):
        """
        :params:
            num_bodys: int; number of bodies to simulate
            mesh_exponent: int; power two exponent for mesh size; example: exponent 8 -> mesh size 256x256
            g_constant: float; gravitational constant
            time_step: float; the base time step, can be manipulated by shower object
            show_size: tuple(2,) of int; size of the output view array
        """
        # initializing moderngl context
        self.context = moderngl.create_context(standalone=True, require=430)

        # setting some constants
        self.num_bodys = num_bodys
        self.g_constant = g_constant
        self.time_step = time_step
        self.mesh_exponent = mesh_exponent
        self.mesh_size = (2**mesh_exponent, 2**mesh_exponent)
        print('Mesh Size:', self.mesh_size)
        self.show_size = show_size

        # loading all compute shaders
        self.program_accumulate = self.context.compute_shader(self.load_file('AccumulationShader.glsl'))
        self.program_step = self.context.compute_shader(self.load_file('StepShader.glsl'))
        self.program_show = self.context.compute_shader(self.load_file('ShowShader.glsl'))
        self.program_fft = self.context.compute_shader(self.load_file('FFTShader.glsl'))
        self.program_fft_input = self.context.compute_shader(self.load_file('FFTInputShader.glsl'))
        self.program_fft_mult = self.context.compute_shader(self.load_file('FFTMultShader.glsl'))
        self.program_fft_output = self.context.compute_shader(self.load_file('FFTOutputShader.glsl'))

        # setting up texure and ssbo buffers
        self.set_up_buffers()
        # setting up FFT convolution filter for force (this takes some time)
        self.set_up_fft_filters()
        # setting up uniforms for the compute shaders
        self.set_up_variables()

        # this line takes all the positions and computes the "orbital" velocity for each object, such that
        # they are in equilibrium from the start, comment this out to have different starting velocities
        #self.set_orbit_velocities()

        # intiializing a color norming factor for smooth transition when zooming
        self.color_norm_factor = 1


    def load_file(self, path):
        """
        helper function for loading shader script
        """
        with open(path, 'r') as f:
            out = f.read()
        return out

    def set_up_buffers(self):
        """
        This function sets up all the necessary GPU buffers for operation
        """
        # this is the initial density array before atomic accumulation
        # (should be zero duh, except you wanna have extra forces)
        self.dens_init = np.zeros(self.mesh_size, dtype=np.float32)
        # uncomment this line to have a very strong gravitational force in the middle
        # self.dens_init[self.mesh_size[0] // 2, self.mesh_size[1] // 2] = 100000000
        self.dens_init = self.dens_init.flatten()
        # texture for the mass density distribution
        self.dens_tex = self.context.texture(self.mesh_size, 1, dtype='f4')
        # texture fore the acceleration
        self.acc_tex = self.context.texture(self.mesh_size, 2, dtype='f4')

        # making the textures periodic
        for tex in [self.dens_tex, self.acc_tex]:
            tex.repeat_x = True
            tex.repeat_y = True

        # the output texture into which the stars are accumulated for visual output
        self.show_init = np.zeros(self.show_size, dtype=np.float32).flatten()
        self.show_tex = self.context.texture(self.show_size, 1, dtype='f4', data=self.show_init)

        # the ssbo buffers for storing information on the bodys
        self.body_buffer = self.context.buffer(reserve=16 * self.num_bodys, dynamic=False)
        self.mass_buffer = self.context.buffer(reserve=4 * self.num_bodys, dynamic=False)
        # here the initial state of the system is set
        body_data, mass_data = self.init_body_data()
        self.body_buffer.write(body_data)
        self.mass_buffer.write(mass_data)

        # buffers for fft stuff (4 components for 2 complex numbers per pixel,
        # one for the x-force and one for the y-force)
        self.dens_fft_tex1 = self.context.texture(self.mesh_size, 4, dtype='f4')
        self.dens_fft_tex2 = self.context.texture(self.mesh_size, 4, dtype='f4')

    def init_body_data(self):
        """
        here, the intiial state is set;
        there are a few examples commented out for random uniform distributions, circles and other stuff
        just write your own initial state here if you want
        :returns:
            body data: np.array(num_bodies, 4); array containing (pos_x, pos_y, vel_x, vel_y) for each body
            mass data: np.array(num_bodies); array containing floats for masses
        """
        body_data = np.zeros((self.num_bodys, 4), dtype=np.float32)
        body_data[:, :2] = np.random.rand(self.num_bodys, 2)**0.5 * 0.3 + 0.3
        body_data[:, 2:] = (body_data[:, :2][:, ::-1] - 0.5) * np.array((-1, 1))[None, :] * 50

        #body_data[:, :2] = np.random.rand(self.num_bodys, 2)
        #body_data[:, 2:] = np.random.randn(self.num_bodys, 2) * 0.1

        
        #simulator = NBodySimulator(10000000, 11, 0.000001, 0.00001, (800, 800))
        #mass_data = (np.abs(np.random.randn(self.num_bodys)) * 10 + 1).astype(np.float32)
        # a disc with more mass at the center
        #r_theta = np.random.rand(self.num_bodys, 2)
        #r_theta[:, 0] = r_theta[:, 0]**2 * 0.2
        #r_theta[1:, 0] += 0.03
        #body_data[:, 0] = r_theta[:, 0] * np.cos(2 * np.pi * r_theta[:, 1]) + 0.5
        #body_data[:, 1] = r_theta[:, 0] * np.sin(2 * np.pi * r_theta[:, 1]) + 0.5

        # initial rotation velocities; these are redundant if set_orbit_velocities() is run
        #v_mag = r_theta[:, 0]**1.2 * 450
        #body_data[:, 2] = -np.sin(2 * np.pi * r_theta[:, 1]) * v_mag
        #body_data[:, 3] = np.cos(2 * np.pi * r_theta[:, 1]) * v_mag
        
        # intiial state for two-bodies for debugging
        #body_data[0, 0] = 0.53
        #body_data[1, 0] = 0.47
        #body_data[:, 1] = 0.5
        #body_data[0, 3] = 1
        #body_data[1, 3] = -1

        #body_data[:, :2] = np.random.rand(self.num_bodys, 2)
        #body_data[:, 2:] = np.random.randn(self.num_bodys, 2) * 0.1

        # mass is abs(random-normal) distributed starting with mass 10
        mass_data = (np.abs(np.random.randn(self.num_bodys)) * 10 + 1).astype(np.float32)

        #mass_data[0] = 20

        # if you wanna have a black hole at the middle:
        #body_data[0, :2] = 0.5
        #body_data[0, 2:] = 0
        #mass_data[0] = 100000000

        return body_data, mass_data

    def set_up_fft_filters(self):
        """
        The method works by using a convolution over the density buffer to get the force field. This convolutional
        filter should contain the x-attraction and y-attraction of a unit mass at the (0, 0) position
        this is computed here and its FFT is computed for the FFT convolution
        """
        # setting up mesh entry positions
        xx, yy = np.indices(self.mesh_size)
        xx = xx / self.mesh_size[0]
        yy = yy / self.mesh_size[1]
        # setting up force arrays
        force_periodic_x = np.zeros((self.mesh_size[0], self.mesh_size[1]))
        force_periodic_y = np.zeros((self.mesh_size[0], self.mesh_size[1]))
        # this loop is done for periodicity (everything more than 3 field-lengths away is neglected)
        for i in range(-3, 3):
            for j in range(-3, 3):
                # standard squared norma distance with a smoothing epsilon
                norm_sq = (xx + i)**2 + (yy + j)**2 + 0.001**2
                norm_sq = np.where(norm_sq == 0, 1, norm_sq)
                force_periodic_x -= (xx + i) / (norm_sq)**(1.5)
                force_periodic_y -= (yy + j) / (norm_sq)**(1.5)

        # this is done to make these properly anti-symmetric
        force_periodic_x -= np.roll(np.roll(force_periodic_x[::-1, ::-1], 1, axis=0), 1, axis=1)
        force_periodic_y -= np.roll(np.roll(force_periodic_y[::-1, ::-1], 1, axis=0), 1, axis=1)
        force_periodic_x *= 0.5
        force_periodic_y *= 0.5
        # we will need the 3x3 neighbourhood around the 0 of the force fields later
        self.force_x_ngbhd = np.roll(np.roll(force_periodic_x, 1, axis=0), 1, axis=1)[:3, :3]
        self.force_y_ngbhd = np.roll(np.roll(force_periodic_y, 1, axis=0), 1, axis=1)[:3, :3]
        # because these are anti-symmetric, the FFT only has a non-zero imaginary part, which we save
        force_x_fft_imag = np.imag(np.fft.fft2(force_periodic_x)) * self.g_constant
        force_y_fft_imag = np.imag(np.fft.fft2(force_periodic_y)) * self.g_constant
        self.forces_fft_imag = np.concatenate((force_x_fft_imag.T[:, :, None], force_y_fft_imag.T[:, :, None]), axis=2).astype(np.float32).copy()

        # the imaginary parts of the x and y force filter ffts are saved to a gpu texture buffer
        self.force_fft_imag_tex = self.context.texture(self.mesh_size, 2, dtype='f4', data=self.forces_fft_imag)

    def set_orbit_velocities(self):
        """
        this function assigns to each particle its orbit velocity around the center
        its slow as its done on the CPU but only has to be done once
        """
        self.dens_tex.write(self.dens_init)

        # accumulate masses into density distribution buffer
        self.body_buffer.bind_to_storage_buffer(0)
        self.mass_buffer.bind_to_storage_buffer(1)
        self.dens_tex.bind_to_image(2)
        self.program_accumulate.run(group_x=int(np.ceil(self.num_bodys / 128)))
        self.context.finish()

        # do the fft convolution thingy to solve for the force
        self.do_fft_convolution()

        # get the force from the GPU buffer
        acc = np.frombuffer(self.acc_tex.read(), 'float32').copy()
        acc = acc.reshape(self.mesh_size[1], self.mesh_size[0], 2).transpose(1, 0, 2)

        # get body positions and velocities
        body_info = np.frombuffer(self.body_buffer.read(), 'float32').copy().reshape(self.num_bodys, 4)
        positions = body_info[:, :2]

        # get acceleration from nearest pixel (some bilinear interpolation would be better but oh well)
        accelerations = acc[(positions[:, 0] * self.mesh_size[0]).astype(np.int), (positions[:, 1] * self.mesh_size[1]).astype(np.int)]

        # get the inward pointing vector and compute the dot product with the acceleration vector
        inward_vec = -(positions - 0.5)
        inward_vec_norm = np.linalg.norm(inward_vec, axis=1)
        inward_vec_norm = np.where(inward_vec_norm > 0, inward_vec_norm, 1)
        inward_vec /= inward_vec_norm[:, None]
        inward_acc = np.maximum(np.sum(accelerations * inward_vec, axis=1), 0)

        # sqrt(acc * r) solves the orbital velocity
        tangential_vel = np.sqrt(inward_acc * inward_vec_norm)

        # aply the tangential vel by using the rotated inward vector
        body_info[:, 2] = inward_vec[:, 1] * tangential_vel
        body_info[:, 3] = -inward_vec[:, 0] * tangential_vel

        # write the new info to the buffer
        self.body_buffer.write(body_info)

    def set_up_variables(self):
        """
        this assigns all uniforms which stay the same throughout the simulation
        """
        self.program_step['mesh_size'] = self.mesh_size
        self.program_step['num_bodys'] = self.num_bodys
        # the stepping shader needs the neighbourhoods to subtract for each particle its own effect on the force field
        # so we dont have auto-gravitation (then the particles buzz around weirdly :D)
        self.program_step['fx_ngbhd'] = tuple(y for x in self.force_x_ngbhd for y in x)
        self.program_step['fy_ngbhd'] = tuple(y for x in self.force_y_ngbhd for y in x)

        self.program_accumulate['mesh_size'] = self.mesh_size
        self.program_accumulate['num_bodys'] = self.num_bodys

        self.program_show['size'] = self.show_size
        self.program_show['num_bodys'] = self.num_bodys

        self.program_fft_input['mesh_exponent'] = self.mesh_exponent
        self.program_fft_mult['mesh_exponent'] = self.mesh_exponent
        self.program_fft_output['mesh_size'] = self.mesh_size

    def make_output_color(self, output_array):
        """
        This function uses a matplotlib colormap to assign colors to the densities
        the coloring is linear for all masses below 100 and logarithmic above that
        """
        output_array = np.where(output_array < 100, output_array / 100, np.log(output_array) - np.log(100) + 1)
        self.color_norm_factor = 0.8 * self.color_norm_factor + 0.2 * np.max(output_array)
        color = cm.plasma(np.minimum(output_array / self.color_norm_factor, 1), bytes=True)[:, :, :3]
        color = np.where(output_array[:, :, None] == 0, 0, color)
        return color

    def do_fft(self):
        """
        helper function which applies the parallel fft using shaders in log time
        """
        # first the fft is done in the x direction
        self.program_fft['x_or_y'] = False
        for level in range(1, self.mesh_exponent + 1):
            # this parameter tells how big the 'blocks' are
            self.program_fft['pow_two_level'] = 2**level
            self.dens_fft_tex1.use(0)
            self.dens_fft_tex2.bind_to_image(1)
            self.program_fft.run(group_x=self.mesh_size[0] // 32, group_y=self.mesh_size[1] // 32)
            self.context.finish()
            self.dens_fft_tex1, self.dens_fft_tex2 = self.dens_fft_tex2, self.dens_fft_tex1

        # then it is done in the y direction
        self.program_fft['x_or_y'] = True
        for level in range(1, self.mesh_exponent + 1):
            self.program_fft['pow_two_level'] = 2**level
            self.dens_fft_tex1.use(0)
            self.dens_fft_tex2.bind_to_image(1)
            self.program_fft.run(group_x=self.mesh_size[0] // 32, group_y=self.mesh_size[1] // 32)
            self.context.finish()
            self.dens_fft_tex1, self.dens_fft_tex2 = self.dens_fft_tex2, self.dens_fft_tex1

    def do_fft_convolution(self):
        """
        handles the whole FFT based convolution of the density distribution with the force filters
        """
        # first the density is reordered by bit-rearranging its indices such that the fft can run in parallel
        self.dens_tex.use(0)
        self.dens_fft_tex2.bind_to_image(1)
        self.program_fft_input.run(group_x=self.mesh_size[0] // 32, group_y=self.mesh_size[1] // 32)
        self.context.finish()
        self.dens_fft_tex1, self.dens_fft_tex2 = self.dens_fft_tex2, self.dens_fft_tex1

        # fft routine is applied on dens_fft_tex
        self.do_fft()

        # now we multiply the fft of our pre-computed force filters in this shader pass
        self.dens_fft_tex1.use(0)
        self.force_fft_imag_tex.use(1)
        self.dens_fft_tex2.bind_to_image(2)
        self.program_fft_mult.run(group_x=self.mesh_size[0] // 32, group_y=self.mesh_size[1] // 32)
        self.context.finish()
        self.dens_fft_tex1, self.dens_fft_tex2 = self.dens_fft_tex2, self.dens_fft_tex1

        # do fft again to as reverse (its the same up do complex conjugation and minus or something like that)
        self.do_fft()

        # get everything into the acceleration buffer by taking the correctly signed imaginary part
        self.dens_fft_tex1.use(0)
        self.acc_tex.bind_to_image(1)
        self.program_fft_output.run(group_x=self.mesh_size[0] // 32, group_y=self.mesh_size[1] // 32)
        self.context.finish()



    def step(self, bbox, time_step_multiplier):
        """
        This is the main solving function, it is executed every step;
        :params:
            bbox: touple(4,) touple containing x, y extent of view box for the view shader
            time_step_multiplier: float; multiplier added to the time_step to speed things up or slow things down
        :returns:
            np.array(show_size, 3) of uint; colored array for display
        """
        # density distribution is initialized to zero to allow atomic accumulation
        self.dens_tex.write(self.dens_init)

        # each object adds its mass to the density distribution at the correct location using imageAtomicAdd
        self.body_buffer.bind_to_storage_buffer(0)
        self.mass_buffer.bind_to_storage_buffer(1)
        self.dens_tex.bind_to_image(2)
        self.program_accumulate.run(group_x=int(np.ceil(self.num_bodys / 128)))
        self.context.finish()

        # the fft method is used to convolve the density distribution with the force filters to get the
        # gravitational acceleration field
        self.do_fft_convolution()

        # each object does one step and updates its velocity according to the acceleration field
        self.program_step['time_step'] = self.time_step * time_step_multiplier
        self.body_buffer.bind_to_storage_buffer(0)
        self.acc_tex.use(2)
        self.program_step.run(group_x=int(np.ceil(self.num_bodys / 128)))
        self.context.finish()

        # the showing buffer is initialized with zero and each object adds its mass to the corresponding pixel
        # using imageAtomicAdd
        self.show_tex.write(self.show_init)
        self.program_show['bbox'] = bbox
        self.body_buffer.bind_to_storage_buffer(0)
        self.mass_buffer.bind_to_storage_buffer(1)
        self.show_tex.bind_to_image(2)
        self.program_show.run(group_x=int(np.ceil(self.num_bodys / 128)))
        self.context.finish()

        # output mass distribution in viewport is transferred CPU, colored and returned
        output_array = np.frombuffer(self.show_tex.read(), 'float32').copy().reshape(self.show_size[1], self.show_size[0]).T
        output_color_array = self.make_output_color(output_array)
        return output_color_array




if __name__ == "__main__":
    simulator = NBodySimulator(10000000, 11, 0.000001, 0.00001, (800, 800))
    shower = Shower(simulator, magnification=1, make_video=True, num_vid_frames=1600)
    shower.mainloop()