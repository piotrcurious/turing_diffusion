/*
 * Enhanced Electrochemical Deposition Visualizer
 * Implements Nernst-Planck transport and Butler-Volmer kinetics
 * 
 * Physics: 
 * 1. Potential Field: Laplace Equation (Secondary Current Distribution)
 * 2. Ion Transport: Nernst-Planck (Diffusion + Migration)
 * 3. Kinetics: Butler-Volmer at the interface
 *
 * Compile: g++ electro_plating_enhanced.cpp -o electro_plating_enhanced -lfltk -lfltk_gl -lGL -lGLU -O3
 */

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/gl.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

const int WIDTH = 300;
const int HEIGHT = 300;

struct Cell {
    double c;      // Concentration of ions (Cu2+)
    double phi;    // Electric potential
    double solid;  // Amount of deposited solid (0.0 to 1.0)
    bool isCathode; // Boundary condition flag
};

class ElectroSim {
public:
    std::vector<Cell> grid;
    std::vector<Cell> nextGrid;
    
    // Physical Constants (Normalized for simulation)
    double D = 0.1;          // Diffusion coefficient
    double mobility = 0.5;   // Ionic mobility
    double alpha = 0.5;      // Transfer coefficient
    double k_rate = 0.1;     // Reaction rate constant (related to j0)
    double voltage = 10.0;   // Applied voltage
    
    ElectroSim() {
        grid.resize(WIDTH * HEIGHT);
        nextGrid.resize(WIDTH * HEIGHT);
        reset();
    }

    void reset() {
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            grid[i].c = 1.0;
            grid[i].phi = 0.0;
            grid[i].solid = 0.0;
            grid[i].isCathode = false;
        }

        // Setup Anode (Top)
        for (int x = 0; x < WIDTH; ++x) {
            // Top boundary acts as source of ions (Anode)
            grid[(HEIGHT - 1) * WIDTH + x].c = 1.0;
        }

        // Setup Initial Cathode (Bottom or Center)
        // Let's do a bottom plate cathode by default
        for (int x = 0; x < WIDTH; x++) {
            grid[0 * WIDTH + x].solid = 1.0;
            grid[0 * WIDTH + x].isCathode = true;
        }
    }

    // Solve Laplace Equation for Potential Field
    void solvePotential() {
        // Successive Over-Relaxation (SOR) for speed
        const double omega = 1.8;
        for (int iter = 0; iter < 10; ++iter) {
            for (int y = 0; y < HEIGHT; ++y) {
                for (int x = 0; x < WIDTH; ++x) {
                    int idx = y * WIDTH + x;
                    
                    // Boundary Conditions
                    if (y == HEIGHT - 1) {
                        grid[idx].phi = voltage; // Anode
                        continue;
                    }
                    if (grid[idx].solid > 0.5) {
                        grid[idx].phi = 0.0; // Cathode (Solid metal is equipotential)
                        continue;
                    }

                    // Finite difference for Laplace
                    int l = (x > 0) ? idx - 1 : idx;
                    int r = (x < WIDTH - 1) ? idx + 1 : idx;
                    int u = (y < HEIGHT - 1) ? idx + WIDTH : idx;
                    int d = (y > 0) ? idx - WIDTH : idx;

                    double target = (grid[l].phi + grid[r].phi + grid[u].phi + grid[d].phi) * 0.25;
                    grid[idx].phi = grid[idx].phi + omega * (target - grid[idx].phi);
                }
            }
        }
    }

    void update() {
        solvePotential();

        for (int y = 1; y < HEIGHT - 1; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                int idx = y * WIDTH + x;
                if (grid[idx].solid > 0.5) continue; // Skip solid cells

                // 1. Diffusion (Laplacian of C)
                int l = (x > 0) ? idx - 1 : (x == 0 ? idx + 1 : idx); // Simple mirror at edges
                int r = (x < WIDTH - 1) ? idx + 1 : (x == WIDTH - 1 ? idx - 1 : idx);
                int u = idx + WIDTH;
                int d = idx - WIDTH;

                double lapC = grid[l].c + grid[r].c + grid[u].c + grid[d].c - 4 * grid[idx].c;

                // 2. Migration (Drift in Potential Field)
                // Flux = - mobility * C * grad(Phi)
                // Div(Flux) = - mobility * (grad(C) * grad(Phi) + C * lap(Phi))
                // Since lap(Phi) = 0 in bulk:
                double dC_dx = (grid[r].c - grid[l].c) * 0.5;
                double dC_dy = (grid[u].c - grid[d].c) * 0.5;
                double dP_dx = (grid[r].phi - grid[l].phi) * 0.5;
                double dP_dy = (grid[u].phi - grid[d].phi) * 0.5;

                double migration = mobility * (dC_dx * dP_dx + dC_dy * dP_dy);

                // 3. Butler-Volmer Kinetics (at the interface)
                double reaction = 0;
                bool isInterface = false;
                // Check neighbors for solid
                if (grid[l].solid > 0.5 || grid[r].solid > 0.5 || grid[u].solid > 0.5 || grid[d].solid > 0.5) {
                    isInterface = true;
                    // Simplified Butler-Volmer: reaction rate proportional to C and exp(-alpha * eta)
                    // Here eta is roughly related to the local potential gradient or just the local Phi
                    double eta = grid[idx].phi - 0.0; // Overpotential (Phi_sol - Phi_metal)
                    reaction = k_rate * grid[idx].c * exp(alpha * eta * 0.5); 
                }

                // Integration
                nextGrid[idx].c = grid[idx].c + D * lapC + migration - reaction;
                nextGrid[idx].solid = grid[idx].solid + reaction * 0.2; // Growth
                
                // Clamp
                if (nextGrid[idx].c < 0) nextGrid[idx].c = 0;
                if (nextGrid[idx].c > 1.2) nextGrid[idx].c = 1.2;
                if (nextGrid[idx].solid > 1.0) nextGrid[idx].solid = 1.0;
            }
        }

        // Update boundaries
        for (int x = 0; x < WIDTH; x++) {
            nextGrid[0 * WIDTH + x] = grid[0 * WIDTH + x];
            nextGrid[(HEIGHT - 1) * WIDTH + x] = grid[(HEIGHT - 1) * WIDTH + x];
            nextGrid[(HEIGHT - 1) * WIDTH + x].c = 1.0; // Source
        }

        grid = nextGrid;
    }
};

class ElectroWindow : public Fl_Gl_Window {
public:
    ElectroSim* sim;
    unsigned char* pixels;

    ElectroWindow(int X, int Y, int W, int H, ElectroSim* s)
        : Fl_Gl_Window(X, Y, W, H), sim(s) {
        pixels = new unsigned char[WIDTH * HEIGHT * 3];
        Fl::add_timeout(1.0/60.0, Timer_CB, (void*)this);
    }

    static void Timer_CB(void* userdata) {
        ElectroWindow* win = (ElectroWindow*)userdata;
        for(int i=0; i<4; i++) win->sim->update();
        win->redraw();
        Fl::repeat_timeout(1.0/60.0, Timer_CB, userdata);
    }

    void draw() {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
        }

        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            double s = sim->grid[i].solid;
            double c = sim->grid[i].c;
            double p = sim->grid[i].phi / 10.0; // Normalized potential for view

            unsigned char r=0, g=0, b=0;

            if (s > 0.1) {
                // Metal Deposit (Copper)
                r = (unsigned char)(180 + s * 75);
                g = (unsigned char)(110 + s * 50);
                b = (unsigned char)(40 + s * 30);
            } else {
                // Solution: Blue (Concentration) + Green (Potential Field)
                r = (unsigned char)(p * 50);
                g = (unsigned char)(p * 100);
                b = (unsigned char)(c * 150);
            }

            pixels[i*3] = r;
            pixels[i*3+1] = g;
            pixels[i*3+2] = b;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0, 0);
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    }
};

void volt_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->voltage = ((Fl_Value_Slider*)w)->value();
}

void reset_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->reset();
}

int main(int argc, char** argv) {
    ElectroSim sim;
    Fl_Double_Window* win = new Fl_Double_Window(500, 400, "Realistic Electroplating Sim");
    ElectroWindow* glWin = new ElectroWindow(10, 10, 300, 300, &sim);

    Fl_Value_Slider* vSlider = new Fl_Value_Slider(320, 30, 150, 20, "Voltage");
    vSlider->bounds(0.0, 50.0);
    vSlider->value(10.0);
    vSlider->callback(volt_cb, &sim);

    Fl_Button* rBtn = new Fl_Button(320, 80, 150, 30, "Reset");
    rBtn->callback(reset_cb, &sim);

    Fl_Box* desc = new Fl_Box(320, 130, 150, 200, 
        "Model:\n- Nernst-Planck\n- Butler-Volmer\n- Laplace Potential\n\n"
        "Colors:\n- Copper: Metal\n- Blue: Ions\n- Green: Potential");
    desc->align(FL_ALIGN_TOP | FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);

    win->end();
    win->show(argc, argv);
    return Fl::run();
}
