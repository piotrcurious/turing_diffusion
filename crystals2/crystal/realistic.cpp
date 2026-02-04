/*
 * Realistic Crystal Growth Visualizer (Turing + Crystal Habit)
 * Features: Real substance presets, Anisotropic growth, Dynamic UI
 *
 * Compile: g++ realistic_crystal.cpp -o realistic_crystal -lfltk -lfltk_gl -lGL -lGLU -O3
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
#include <string>
#include <cmath>
#include <iostream>

// --- Physics Constants ---
const int WIDTH = 300;
const int HEIGHT = 300;

// Structure to define physical properties of a mineral
struct Substance {
    std::string name;
    double diffU;      // Diffusion of Activator (Solute)
    double diffV;      // Diffusion of Inhibitor (Heat/Depletion)
    double F;          // Feed rate (roughly correlates to supersaturation)
    double k;          // Kill rate (roughly correlates to solubility limit)
    double anisotropy; // 0.0 = Isotropic, 1.0 = Highly directional lattice
    float r, g, b;     // Natural color of the mineral
};

// Global Simulation State
std::vector<Substance> substances;
Substance currentSubstance;
bool resetFlag = false;

// UI Pointers
Fl_Value_Slider* fSliderPtr = nullptr;
Fl_Value_Slider* kSliderPtr = nullptr;

// --- Physics Engine ---

struct Cell {
    double u; // Concentration of Solute A
    double v; // Concentration of Solute B (Precipitate)
};

class CrystalSimulation {
public:
    std::vector<Cell> grid;
    std::vector<Cell> nextGrid;

    CrystalSimulation() {
        grid.resize(WIDTH * HEIGHT);
        nextGrid.resize(WIDTH * HEIGHT);
        // Define Real-ish presets based on pattern morphology
        
        // 1. Copper Sulfate (Triclinic): Chaotic, blue, fast growth
        substances.push_back({"Copper Sulfate", 0.16, 0.08, 0.060, 0.062, 0.05, 0.0f, 0.4f, 0.9f});
        
        // 2. Snow/Ice (Hexagonal): Very low feed, high symmetry, dendritic
        substances.push_back({"Ice (Snowflake)", 0.19, 0.09, 0.035, 0.060, 0.80, 0.9f, 0.95f, 1.0f});
        
        // 3. Pyrite (Cubic): Blocky, gold, stable patterns
        substances.push_back({"Pyrite (Fools Gold)", 0.14, 0.06, 0.040, 0.060, 0.95, 0.8f, 0.7f, 0.2f});

        // 4. Malachite (Botryoidal): Banded, green, oscillating
        substances.push_back({"Malachite", 0.12, 0.08, 0.025, 0.055, 0.10, 0.1f, 0.8f, 0.3f});

        currentSubstance = substances[0];
        reset();
    }

    void reset() {
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            grid[i].u = 1.0;
            grid[i].v = 0.0;
        }
        seed(WIDTH / 2, HEIGHT / 2, 5);
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

    // Anisotropic Laplacian (Simulates Crystal Lattice Bias)
    void getLaplacian(int x, int y, double& lapU, double& lapV) {
        double u = grid[y * WIDTH + x].u;
        double v = grid[y * WIDTH + x].v;

        // Orthogonal neighbors
        int l = (x - 1 + WIDTH) % WIDTH;
        int r = (x + 1) % WIDTH;
        int u_y = (y - 1 + HEIGHT) % HEIGHT;
        int d = (y + 1) % HEIGHT;

        // Weights
        // If anisotropy is high, diagonals count for less (simulating cubic/grid locking)
        double diagWeight = 0.05 * (1.0 - currentSubstance.anisotropy); 
        double orthoWeight = 0.2 + (0.05 * currentSubstance.anisotropy); // Redistribution of weight

        double sumU = 0.0;
        double sumV = 0.0;

        // Orthogonal
        sumU += (grid[y * WIDTH + l].u + grid[y * WIDTH + r].u + grid[u_y * WIDTH + x].u + grid[d * WIDTH + x].u) * orthoWeight;
        sumV += (grid[y * WIDTH + l].v + grid[y * WIDTH + r].v + grid[u_y * WIDTH + x].v + grid[d * WIDTH + x].v) * orthoWeight;

        // Diagonals
        sumU += (grid[u_y * WIDTH + l].u + grid[u_y * WIDTH + r].u + grid[d * WIDTH + l].u + grid[d * WIDTH + r].u) * diagWeight;
        sumV += (grid[u_y * WIDTH + l].v + grid[u_y * WIDTH + r].v + grid[d * WIDTH + l].v + grid[d * WIDTH + r].v) * diagWeight;

        // Center
        sumU -= u;
        sumV -= v;

        lapU = sumU;
        lapV = sumV;
    }

    void update() {
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            int y = i / WIDTH;
            int x = i % WIDTH;

            double u = grid[i].u;
            double v = grid[i].v;

            double lapU, lapV;
            getLaplacian(x, y, lapU, lapV);

            double reaction = u * v * v;
            
            // Standard Gray-Scott Reaction-Diffusion
            double du = (currentSubstance.diffU * lapU) - reaction + (currentSubstance.F * (1.0 - u));
            double dv = (currentSubstance.diffV * lapV) + reaction - ((currentSubstance.F + currentSubstance.k) * v);

            double newU = u + du;
            double newV = v + dv;

            // Clamp
            if(newU < 0) newU = 0; if(newU > 1) newU = 1;
            if(newV < 0) newV = 0; if(newV > 1) newV = 1;

            nextGrid[i].u = newU;
            nextGrid[i].v = newV;
        }
        grid = nextGrid;
    }
};

// --- Visualization Window ---

class SimulationView : public Fl_Gl_Window {
public:
    CrystalSimulation* sim;
    unsigned char* pixels;

    SimulationView(int X, int Y, int W, int H, CrystalSimulation* s)
        : Fl_Gl_Window(X, Y, W, H), sim(s) {
        pixels = new unsigned char[WIDTH * HEIGHT * 3];
        Fl::add_timeout(1.0/60.0, Timer_CB, (void*)this);
    }

    static void Timer_CB(void* userdata) {
        SimulationView* win = (SimulationView*)userdata;
        if(resetFlag) {
            win->sim->reset();
            resetFlag = false;
        }
        // Run multiple steps for speed
        for(int i=0; i<8; i++) win->sim->update();
        win->redraw();
        Fl::repeat_timeout(1.0/60.0, Timer_CB, userdata);
    }

    void draw() {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
        }

        // Render Loop
        for (int i = 0; i < WIDTH * HEIGHT; ++i) {
            double v = sim->grid[i].v; // Inhibitor represents the solid crystal
            
            // Visualization: Map density to substance color
            // v usually ranges 0.0 to ~0.6 in these sims
            double intensity = v * 3.5; 
            if(intensity > 1.0) intensity = 1.0;
            
            // Background is dark (solution)
            // Crystal is colored
            float r = currentSubstance.r * intensity;
            float g = currentSubstance.g * intensity;
            float b = currentSubstance.b * intensity;

            // Highlight the "Growth Edge" (high reaction zone)
            if(v > 0.2 && v < 0.4) {
                r += 0.2; g += 0.2; b += 0.2; // Shine effect
            }

            pixels[i*3]   = (unsigned char)(r * 255);
            pixels[i*3+1] = (unsigned char)(g * 255);
            pixels[i*3+2] = (unsigned char)(b * 255);
        }

        glClearColor(0.1, 0.1, 0.1, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0, 0);
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    }

    int handle(int event) {
        if (event == FL_PUSH || event == FL_DRAG) {
            // Mapping mouse to grid
            double sx = (double)WIDTH / w();
            double sy = (double)HEIGHT / h();
            int gx = Fl::event_x() * sx;
            int gy = (h() - Fl::event_y()) * sy;
            sim->seed(gx, gy, 4);
            return 1;
        }
        return Fl_Gl_Window::handle(event);
    }
};

// --- UI Callbacks ---

void choice_cb(Fl_Widget* w, void* v) {
    Fl_Choice* c = (Fl_Choice*)w;
    int idx = c->value();
    currentSubstance = substances[idx];
    
    // Update Sliders to match preset
    if (fSliderPtr) fSliderPtr->value(currentSubstance.F);
    if (kSliderPtr) kSliderPtr->value(currentSubstance.k);

    std::cout << "Selected: " << currentSubstance.name << std::endl;
    resetFlag = true;
}

void feed_cb(Fl_Widget* w, void* v) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    currentSubstance.F = s->value();
}

void kill_cb(Fl_Widget* w, void* v) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    currentSubstance.k = s->value();
}

void reset_cb(Fl_Widget* w, void* v) {
    resetFlag = true;
}

// --- Main ---

int main(int argc, char** argv) {
    CrystalSimulation sim;

    Fl_Double_Window* win = new Fl_Double_Window(600, 450, "Advanced Crystal Growth Lab");
    
    // Left: Simulation View
    SimulationView* simView = new SimulationView(10, 10, 400, 400, &sim);
    
    // Right: Controls
    Fl_Choice* choice = new Fl_Choice(430, 30, 150, 25, "Substance");
    for(size_t i=0; i<substances.size(); i++) {
        choice->add(substances[i].name.c_str());
    }
    choice->value(0);
    choice->callback(choice_cb);
    
    fSliderPtr = new Fl_Value_Slider(430, 100, 150, 20, "Feed (F) - Supply");
    fSliderPtr->type(FL_HOR_NICE_SLIDER);
    fSliderPtr->bounds(0.01, 0.1);
    fSliderPtr->value(substances[0].F);
    fSliderPtr->callback(feed_cb);
    
    kSliderPtr = new Fl_Value_Slider(430, 160, 150, 20, "Kill (k) - Solubility");
    kSliderPtr->type(FL_HOR_NICE_SLIDER);
    kSliderPtr->bounds(0.03, 0.07);
    kSliderPtr->value(substances[0].k);
    kSliderPtr->callback(kill_cb);

    Fl_Button* btnReset = new Fl_Button(430, 380, 150, 30, "Clear / Reset");
    btnReset->callback(reset_cb);

    // Instructions
    Fl_Box* note = new Fl_Box(430, 200, 150, 150, "Controls:\n\nSelect a mineral preset\nto see different\ngrowth habits.\n\nClick in the black box\nto seed crystals.");
    note->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE);

    win->end();
    win->show(argc, argv);
    return Fl::run();
}
