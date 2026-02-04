/*
 * Crystal Growth Visualizer (Turing Reaction-Diffusion)
 * Based on concepts of Morphogenesis and Crystal Habit
 *
 * Libraries: FLTK, OpenGL
 * Compile: g++ crystal_growth.cpp -o crystal_growth -lfltk -lfltk_gl -lGL -lGLU -O3
 */

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <vector>
#include <cmath>
#include <iostream>

// --- Simulation Parameters (Section 1 & 4) ---
// Grid dimensions
const int WIDTH = 256;
const int HEIGHT = 256;

// Diffusion Rates (Du, Dv) - The key to Turing Instability
// Changing these alters the "crystal habit" (Section 3)
const double diffU = 0.16;
const double diffV = 0.08;

// Reaction parameters (Feed and Kill rates)
// These represent environmental control (Temperature/Concentration)
// "Crystal" regime often found around F=0.060, k=0.062
const double F = 0.060; 
const double k = 0.062; 

struct Cell {
    double u; // Activator concentration (e.g., Ion availability)
    double v; // Inhibitor concentration (e.g., Depletion zone)
};

class CrystalSimulation {
public:
    std::vector<Cell> grid;
    std::vector<Cell> nextGrid;

    CrystalSimulation() {
        grid.resize(WIDTH * HEIGHT);
        nextGrid.resize(WIDTH * HEIGHT);
        reset();
    }

    // Initialize the homogeneous state with slight "Supersaturation"
    void reset() {
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            grid[i].u = 1.0;
            grid[i].v = 0.0;
        }
        // Create a central nucleation site (Section 2)
        seed(WIDTH / 2, HEIGHT / 2, 10);
    }

    // Introduce local disturbance (Nucleation / Impurity)
    void seed(int x, int y, int radius) {
        for (int i = x - radius; i < x + radius; ++i) {
            for (int j = y - radius; j < y + radius; ++j) {
                if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT) {
                    // Inject inhibitor/convert state to start reaction
                    grid[j * WIDTH + i].v = 0.5; 
                }
            }
        }
    }

    // Laplacian using a 3x3 convolution kernel
    // Represents the diffusion operator (del squared) in the PDEs
    void getLaplacian(int x, int y, double& lapU, double& lapV) {
        double u = grid[y * WIDTH + x].u;
        double v = grid[y * WIDTH + x].v;

        double sumU = 0.0;
        double sumV = 0.0;

        // Simple convolution weights for diffusion
        // Center: -1, Neighbors: 0.2, Diagonals: 0.05
        
        // Orthogonal neighbors
        int l = (x - 1 + WIDTH) % WIDTH;
        int r = (x + 1) % WIDTH;
        int u_y = (y - 1 + HEIGHT) % HEIGHT;
        int d = (y + 1) % HEIGHT;

        sumU += grid[y * WIDTH + l].u * 0.2;
        sumU += grid[y * WIDTH + r].u * 0.2;
        sumU += grid[u_y * WIDTH + x].u * 0.2;
        sumU += grid[d * WIDTH + x].u * 0.2;

        sumV += grid[y * WIDTH + l].v * 0.2;
        sumV += grid[y * WIDTH + r].v * 0.2;
        sumV += grid[u_y * WIDTH + x].v * 0.2;
        sumV += grid[d * WIDTH + x].v * 0.2;

        // Diagonals
        sumU += grid[u_y * WIDTH + l].u * 0.05;
        sumU += grid[u_y * WIDTH + r].u * 0.05;
        sumU += grid[d * WIDTH + l].u * 0.05;
        sumU += grid[d * WIDTH + r].u * 0.05;
        sumV += grid[u_y * WIDTH + l].v * 0.05;
        sumV += grid[u_y * WIDTH + r].v * 0.05;
        sumV += grid[d * WIDTH + l].v * 0.05;
        sumV += grid[d * WIDTH + r].v * 0.05;

        // Center weight (-1)
        sumU += u * -1.0;
        sumV += v * -1.0;

        lapU = sumU;
        lapV = sumV;
    }

    // The Reaction-Diffusion Step (Gray-Scott Model)
    // Matches the equation form: du/dt = Du*Laplace(u) - uv^2 + F(1-u)
    void update() {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                double u = grid[y * WIDTH + x].u;
                double v = grid[y * WIDTH + x].v;

                double lapU, lapV;
                getLaplacian(x, y, lapU, lapV);

                // Reaction term uv^2
                double reaction = u * v * v;

                // Update equations
                double du = (diffU * lapU) - reaction + (F * (1.0 - u));
                double dv = (diffV * lapV) + reaction - ((F + k) * v);

                double newU = u + du;
                double newV = v + dv;

                // Clamp values
                if(newU < 0) newU = 0; if(newU > 1) newU = 1;
                if(newV < 0) newV = 0; if(newV > 1) newV = 1;

                nextGrid[y * WIDTH + x].u = newU;
                nextGrid[y * WIDTH + x].v = newV;
            }
        }
        // Swap buffers
        grid = nextGrid;
    }
};

class CrystalWindow : public Fl_Gl_Window {
    CrystalSimulation sim;
    unsigned char* pixels; // Texture data

public:
    CrystalWindow(int X, int Y, int W, int H, const char* L)
        : Fl_Gl_Window(X, Y, W, H, L) {
        pixels = new unsigned char[WIDTH * HEIGHT * 3];
        Fl::add_timeout(1.0/60.0, Timer_CB, (void*)this);
    }

    ~CrystalWindow() {
        delete[] pixels;
    }

    static void Timer_CB(void* userdata) {
        CrystalWindow* win = (CrystalWindow*)userdata;
        
        // Speed up simulation by running multiple steps per frame
        for(int i=0; i<12; i++) {
            win->sim.update();
        }
        
        win->redraw();
        Fl::repeat_timeout(1.0/60.0, Timer_CB, userdata);
    }

    // Interaction: Click to add "Impurities" or Nucleation sites
    int handle(int event) {
        if (event == FL_PUSH || event == FL_DRAG) {
            int mx = Fl::event_x();
            int my = Fl::event_y();
            
            // Map window coords to grid coords
            float scaleX = (float)WIDTH / w();
            float scaleY = (float)HEIGHT / h();
            
            // OpenGL coords are inverted in Y relative to window
            int gridX = mx * scaleX;
            int gridY = (h() - my) * scaleY;

            sim.seed(gridX, gridY, 5);
            return 1;
        }
        return Fl_Gl_Window::handle(event);
    }

    void draw() {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
        }

        // Convert Simulation Grid to Colors (Visualization)
        // High Activator (u) - Low Inhibitor (v) = Solid Crystal
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            double u = sim.grid[i].u;
            double v = sim.grid[i].v;

            // Visualization Logic:
            // We map the concentrations to a "Mineral" color palette.
            // Dark regions = Solution
            // White/Cyan regions = Crystal Structure
            
            unsigned char r, g, b;
            
            // Simple mapping: visualize the "inhibitor" v as the crystal structure
            // because in Gray-Scott, v forms the patterns.
            double val = v * 3.0; // Scale for visibility
            if(val > 1.0) val = 1.0;

            r = (unsigned char)(val * 20);   // Dark mineral base
            g = (unsigned char)(val * 200);  // Cyan/Greenish tint
            b = (unsigned char)(val * 255);  // Blue dominance

            // Add a "shine" for high concentration (crystallization front)
            if (v > 0.3) {
                r = 200; g = 220; b = 255;
            }

            pixels[i*3] = r;
            pixels[i*3+1] = g;
            pixels[i*3+2] = b;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        
        // Draw the pixel array directly to screen
        glRasterPos2i(0, 0);
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    }
};

int main(int argc, char** argv) {
    Fl_Window* mainWin = new Fl_Window(512, 512, "Turing Crystal Growth Visualizer");
    CrystalWindow* simWin = new CrystalWindow(0, 0, 512, 512, "Sim");
    
    mainWin->resizable(simWin);
    mainWin->end();
    mainWin->show(argc, argv);
    
    std::cout << "--- Controls ---\n";
    std::cout << "Mouse Click/Drag: Introduce nucleation sites (impurities)\n";
    std::cout << "Watch as Turing Diffusion creates dendritic crystal patterns.\n";
    
    return Fl::run();
}
