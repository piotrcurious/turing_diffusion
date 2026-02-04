/*
 * Electrochemical Crystal Growth Visualizer
 * Combines Turing Reaction-Diffusion with Nernst-Planck Electromigration
 *
 * Physics: du/dt = D*Laplacian(u) - div(Flux_drift) + Reaction(u,v)
 * Flux_drift = mobility * u * E_field
 *
 * Compile: g++ electro_crystal.cpp -o electro_crystal -lfltk -lfltk_gl -lGL -lGLU -O3
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

// --- Simulation Constants ---
const int WIDTH = 300;
const int HEIGHT = 300;

// Enum for Electric Field Configuration
enum FieldType {
    FIELD_NONE = 0,
    FIELD_LINEAR_DOWN = 1, // Parallel Plate (Top Anode -> Bottom Cathode)
    FIELD_RADIAL_IN = 2    // Point Cathode in Center (Wire growth)
};

// Chemical Properties (using the "Copper Sulfate" base for best dendritic visualization)
struct ChemPhysics {
    double diffU = 0.16;   // Cation Diffusion
    double diffV = 0.08;   // Crystal Complex Diffusion
    double F = 0.060;      // Feed Rate
    double k = 0.062;      // Kill Rate
};

// Global State
ChemPhysics physics;
double voltage = 0.0;     // Strength of Electric Field (0.0 to 2.0)
int fieldType = FIELD_NONE;
bool resetFlag = false;

struct Cell {
    double u; // Cation Concentration (+)
    double v; // Precipitate / Deposition
};

class ElectroSimulation {
public:
    std::vector<Cell> grid;
    std::vector<Cell> nextGrid;

    ElectroSimulation() {
        grid.resize(WIDTH * HEIGHT);
        nextGrid.resize(WIDTH * HEIGHT);
        reset();
    }

    void reset() {
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            grid[i].u = 1.0;
            grid[i].v = 0.0;
        }
        
        // Initial seeding depends on field type for best effect
        if (fieldType == FIELD_RADIAL_IN) {
            // Seed a "Wire" in the center
            seed(WIDTH/2, HEIGHT/2, 4);
        } else if (fieldType == FIELD_LINEAR_DOWN) {
            // Seed the "Cathode Plate" at the bottom
            for(int x=0; x<WIDTH; x+=10) seed(x, HEIGHT-5, 2);
        } else {
            // Random blob
            seed(WIDTH/2, HEIGHT/2, 10);
        }
    }

    void seed(int x, int y, int radius) {
        for (int i = x - radius; i < x + radius; ++i) {
            for (int j = y - radius; j < y + radius; ++j) {
                if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT) {
                    grid[j * WIDTH + i].v = 0.5;
                }
            }
        }
    }

    // Calculate Electric Field Vector at grid point (x,y)
    void getElectricField(int x, int y, double& Ex, double& Ey) {
        if (fieldType == FIELD_NONE || voltage == 0.0) {
            Ex = 0; Ey = 0;
            return;
        }

        if (fieldType == FIELD_LINEAR_DOWN) {
            // Field points DOWN (driving cations down to cathode)
            Ex = 0.0;
            Ey = voltage * 0.5; 
        } 
        else if (fieldType == FIELD_RADIAL_IN) {
            // Field points to Center
            double dx = (WIDTH / 2.0) - x;
            double dy = (HEIGHT / 2.0) - y;
            double dist = sqrt(dx*dx + dy*dy) + 0.1; // Avoid div/0
            
            // Field gets stronger near the electrode (1/r^2 usually, but 1/r is cleaner visually)
            double strength = (voltage * 10.0) / dist;
            
            Ex = (dx / dist) * strength;
            Ey = (dy / dist) * strength;
        }
    }

    void update() {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                int idx = y * WIDTH + x;
                double u = grid[idx].u;
                double v = grid[idx].v;

                // 1. Diffusion (Laplacian)
                // Using simplified 5-point stencil for speed/readability here
                double sumU = 0, sumV = 0;
                int l = (x - 1 + WIDTH) % WIDTH;
                int r = (x + 1) % WIDTH;
                int u_y = (y - 1 + HEIGHT) % HEIGHT;
                int d = (y + 1) % HEIGHT;

                sumU = grid[y*WIDTH + l].u + grid[y*WIDTH + r].u + grid[u_y*WIDTH + x].u + grid[d*WIDTH + x].u - 4*u;
                sumV = grid[y*WIDTH + l].v + grid[y*WIDTH + r].v + grid[u_y*WIDTH + x].v + grid[d*WIDTH + x].v - 4*v;

                // 2. Electromigration (Advection)
                // Transport u (cations) based on Electric Field
                // Advection term: - (v_x * du/dx + v_y * du/dy)
                double advU = 0.0;
                double Ex, Ey;
                getElectricField(x, y, Ex, Ey);

                if (abs(Ex) > 0 || abs(Ey) > 0) {
                    // Upwind Difference Scheme for stability
                    // If moving right (Ex > 0), info comes from Left
                    double du_dx = 0;
                    if (Ex > 0) du_dx = u - grid[y*WIDTH + l].u;
                    else        du_dx = grid[y*WIDTH + r].u - u;

                    double du_dy = 0;
                    if (Ey > 0) du_dy = u - grid[u_y*WIDTH + x].u;
                    else        du_dy = grid[d*WIDTH + x].u - u;

                    advU = - (Ex * du_dx + Ey * du_dy);
                }

                // 3. Reaction (Gray-Scott)
                double reaction = u * v * v;
                
                // 4. Time Step Integration
                double du = (physics.diffU * sumU) + advU - reaction + (physics.F * (1.0 - u));
                double dv = (physics.diffV * sumV) + reaction - ((physics.F + physics.k) * v);

                // Note: Inhibitor 'v' (the crystal) is solid, so it does not drift (advV = 0)
                // Only 'u' (the ions in solution) drift.

                double newU = u + du;
                double newV = v + dv;

                // Clamp
                if(newU < 0) newU = 0; if(newU > 1) newU = 1;
                if(newV < 0) newV = 0; if(newV > 1) newV = 1;

                nextGrid[idx].u = newU;
                nextGrid[idx].v = newV;
            }
        }
        grid = nextGrid;
    }
};

class ElectroWindow : public Fl_Gl_Window {
public:
    ElectroSimulation* sim;
    unsigned char* pixels;

    ElectroWindow(int X, int Y, int W, int H, ElectroSimulation* s)
        : Fl_Gl_Window(X, Y, W, H), sim(s) {
        pixels = new unsigned char[WIDTH * HEIGHT * 3];
        Fl::add_timeout(1.0/60.0, Timer_CB, (void*)this);
    }

    static void Timer_CB(void* userdata) {
        ElectroWindow* win = (ElectroWindow*)userdata;
        if(resetFlag) {
            win->sim->reset();
            resetFlag = false;
        }
        for(int i=0; i<6; i++) win->sim->update();
        win->redraw();
        Fl::repeat_timeout(1.0/60.0, Timer_CB, userdata);
    }

    void draw() {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
        }

        // Visualize
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            double v = sim->grid[i].v;
            double u = sim->grid[i].u;

            // Color Map: 
            // Crystal (v) = Copper/Gold colors
            // Solution (u) = Faint blue trace
            
            unsigned char r=0, g=0, b=0;

            if (v > 0.1) {
                // Crystal Structure
                double intensity = v * 4.0;
                if(intensity > 1) intensity = 1;
                
                // Electric Copper Look
                r = (unsigned char)(200 * intensity);
                g = (unsigned char)(100 * intensity);
                b = (unsigned char)(50 * intensity);
            } else {
                // Visualization of the Ionic Cloud (u)
                // This lets us see the "Drift" visually!
                double ion = (1.0 - u) * 2.0; // Invert: where u is depleted
                if(ion < 0) ion = 0;
                b = (unsigned char)(ion * 50);
            }

            pixels[i*3] = r;
            pixels[i*3+1] = g;
            pixels[i*3+2] = b;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0, 0);
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        
        // Draw Field Direction Arrow (Overlay)
        if(voltage > 0.0) {
            glColor3f(1.0, 1.0, 0.0);
            glBegin(GL_LINES);
            if(fieldType == FIELD_LINEAR_DOWN) {
                // Down arrow in corner
                glVertex2i(20, HEIGHT-20); glVertex2i(20, HEIGHT-50);
                glVertex2i(20, HEIGHT-50); glVertex2i(15, HEIGHT-40);
                glVertex2i(20, HEIGHT-50); glVertex2i(25, HEIGHT-40);
            }
            if(fieldType == FIELD_RADIAL_IN) {
                // X in corner
                glVertex2i(10, HEIGHT-10); glVertex2i(30, HEIGHT-30);
                glVertex2i(10, HEIGHT-30); glVertex2i(30, HEIGHT-10);
            }
            glEnd();
        }
    }
    
    int handle(int event) {
        if (event == FL_PUSH || event == FL_DRAG) {
            double sx = (double)WIDTH / w();
            double sy = (double)HEIGHT / h();
            sim->seed(Fl::event_x() * sx, (h() - Fl::event_y()) * sy, 4);
            return 1;
        }
        return Fl_Gl_Window::handle(event);
    }
};

// --- Callbacks ---
void field_cb(Fl_Widget* w, void* v) {
    Fl_Choice* c = (Fl_Choice*)w;
    fieldType = c->value();
    resetFlag = true;
}

void volt_cb(Fl_Widget* w, void* v) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    voltage = s->value();
}

void reset_cb(Fl_Widget* w, void* v) {
    resetFlag = true;
}

int main(int argc, char** argv) {
    ElectroSimulation sim;
    Fl_Double_Window* win = new Fl_Double_Window(550, 420, "Electrochemical Growth Lab");

    ElectroWindow* glWin = new ElectroWindow(10, 10, 350, 400, &sim);
    
    // Controls
    Fl_Group* controls = new Fl_Group(370, 10, 170, 400);
    
    Fl_Choice* fChoice = new Fl_Choice(370, 30, 170, 25, "Field Topology");
    fChoice->add("None (Pure Diffusion)");
    fChoice->add("Linear (Parallel Plate)");
    fChoice->add("Radial (Point Wire)");
    fChoice->value(0);
    fChoice->callback(field_cb);
    
    Fl_Value_Slider* vSlider = new Fl_Value_Slider(370, 90, 170, 20, "Voltage (Drift Force)");
    vSlider->type(FL_HOR_NICE_SLIDER);
    vSlider->bounds(0.0, 1.0);
    vSlider->value(0.0);
    vSlider->callback(volt_cb);
    
    Fl_Box* info = new Fl_Box(370, 130, 170, 150, 
        "Info:\nUse Voltage to push\nions into the crystal.\n\n"
        "Linear: Grows down.\nRadial: Grows inward.\n\n"
        "Simulates Electroplating\nand Dendrite formation.");
    info->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE);
    
    Fl_Button* rBtn = new Fl_Button(370, 350, 170, 30, "Reset Simulation");
    rBtn->callback(reset_cb);
    
    controls->end();
    win->end();
    win->show(argc, argv);
    return Fl::run();
}
