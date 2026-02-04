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
    int symmetry;      // 0: Isotropic, 4: Cubic, 6: Hexagonal
    float r, g, b;     // Natural color of the mineral
    std::string info;  // Molecular structure explanation
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
        substances.push_back({"Copper Sulfate", 0.16, 0.08, 0.060, 0.062, 0.05, 0, 0.0f, 0.4f, 0.9f,
            "Structure: Triclinic\nLow symmetry leads to chaotic, branching growth."});
        
        // 2. Snow/Ice (Hexagonal): Very low feed, high symmetry, dendritic
        substances.push_back({"Ice (Snowflake)", 0.19, 0.09, 0.035, 0.060, 0.80, 6, 0.9f, 0.95f, 1.0f,
            "Structure: Hexagonal (Ih)\n6-fold symmetry from hydrogen bond networks."});
        
        // 3. Pyrite (Cubic): Blocky, gold, stable patterns
        substances.push_back({"Pyrite (Fools Gold)", 0.14, 0.06, 0.040, 0.060, 0.95, 4, 0.8f, 0.7f, 0.2f,
            "Structure: Cubic\nStrong axis-aligned growth creates blocky crystals."});

        // 4. Malachite (Botryoidal): Banded, green, oscillating
        substances.push_back({"Malachite", 0.12, 0.08, 0.025, 0.055, 0.10, 0, 0.1f, 0.8f, 0.3f,
            "Structure: Monoclinic\nForms botryoidal (grape-like) masses."});

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

        // Add symmetry bias based on angle from center
        double angle = atan2(y - HEIGHT/2.0, x - WIDTH/2.0);
        double bias = 1.0;
        if (currentSubstance.symmetry > 0) {
            bias = 1.0 + currentSubstance.anisotropy * cos(currentSubstance.symmetry * angle);
        }

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
    bool view3d = false;
    float rotX = 20, rotY = -20;

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
            glEnable(GL_DEPTH_TEST);
        }

        if (view3d) {
            draw3d();
            return;
        }

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

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

    void draw3d() {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (double)w()/h(), 1, 1000);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0, 0, -400);
        glRotatef(rotX, 1, 0, 0);
        glRotatef(rotY, 0, 1, 0);
        glTranslatef(-WIDTH/2, -HEIGHT/2, 0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBegin(GL_POINTS);
        for (int y = 0; y < HEIGHT; y+=2) {
            for (int x = 0; x < WIDTH; x+=2) {
                int i = y * WIDTH + x;
                double v = sim->grid[i].v;
                if (v < 0.01) continue;

                double intensity = v * 3.5;
                if(intensity > 1.0) intensity = 1.0;

                glColor3f(currentSubstance.r * intensity,
                          currentSubstance.g * intensity,
                          currentSubstance.b * intensity);

                glVertex3f(x, y, v * 100.0);
            }
        }
        glEnd();
    }

    int handle(int event) {
        static int last_mx, last_my;
        if (event == FL_PUSH) {
            last_mx = Fl::event_x();
            last_my = Fl::event_y();
        }

        if (event == FL_DRAG) {
            if (Fl::event_button3() || (Fl::event_state() & FL_SHIFT)) {
                rotY += (Fl::event_x() - last_mx);
                rotX += (Fl::event_y() - last_my);
                last_mx = Fl::event_x();
                last_my = Fl::event_y();
                redraw();
                return 1;
            }
        }

        if (event == FL_PUSH || event == FL_DRAG) {
            if (!Fl::event_button3() && !(Fl::event_state() & FL_SHIFT)) {
                // Mapping mouse to grid
                int mx = Fl::event_x() - x();
                int my = Fl::event_y() - y();
                double sx = (double)WIDTH / w();
                double sy = (double)HEIGHT / h();
                int gx = mx * sx;
                int gy = (h() - my) * sy;
                sim->seed(gx, gy, 4);
                return 1;
            }
        }
        return Fl_Gl_Window::handle(event);
    }
};

// UI Pointers
Fl_Box* notePtr = nullptr;

// --- UI Callbacks ---

void view_cb(Fl_Widget* w, void* v) {
    SimulationView* view = (SimulationView*)v;
    view->view3d = !view->view3d;
    view->redraw();
}

void choice_cb(Fl_Widget* w, void* v) {
    Fl_Choice* c = (Fl_Choice*)w;
    int idx = c->value();
    currentSubstance = substances[idx];
    
    // Update Sliders to match preset
    if (fSliderPtr) fSliderPtr->value(currentSubstance.F);
    if (kSliderPtr) kSliderPtr->value(currentSubstance.k);
    if (notePtr) notePtr->label(currentSubstance.info.c_str());

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

    Fl_Button* btn3d = new Fl_Button(430, 200, 150, 30, "Toggle 3D View");
    btn3d->callback(view_cb, simView);

    // Instructions / Molecular Info
    notePtr = new Fl_Box(430, 240, 150, 100, currentSubstance.info.c_str());
    notePtr->align(FL_ALIGN_WRAP | FL_ALIGN_TOP | FL_ALIGN_INSIDE);

    Fl_Button* btnReset = new Fl_Button(430, 380, 150, 30, "Clear / Reset");
    btnReset->callback(reset_cb);

    win->end();
    win->show(argc, argv);
    return Fl::run();
}
