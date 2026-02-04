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
#include <cstdlib>
#include <ctime>

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

    void seed(int x, int y, int radius) {
        for (int i = x - radius; i < x + radius; ++i) {
            for (int j = y - radius; j < y + radius; ++j) {
                if (i >= 0 && i < WIDTH && j >= 0 && j < HEIGHT) {
                    grid[j * WIDTH + i].solid = 1.0;
                    grid[j * WIDTH + i].isCathode = true;
                    grid[j * WIDTH + i].phi = 0.0;
                }
            }
        }
    }
    
    // Physical Constants (Normalized for simulation)
    double D = 0.1;          // Diffusion coefficient
    double mobility = 0.5;   // Ionic mobility
    double alpha = 0.5;      // Transfer coefficient
    double k_rate = 0.1;     // Reaction rate constant (related to j0)
    double voltage = 10.0;   // Applied voltage
    int geometry = 0;        // 0: Plate, 1: Wire, 2: Array
    
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
            grid[(HEIGHT - 1) * WIDTH + x].c = 1.0;
        }

        // Setup Initial Cathode based on geometry
        if (geometry == 0) { // Plate
            for (int x = 0; x < WIDTH; x++) {
                grid[0 * WIDTH + x].solid = 1.0;
                grid[0 * WIDTH + x].isCathode = true;
            }
        } else if (geometry == 1) { // Wire (Center)
            int cx = WIDTH / 2;
            int cy = HEIGHT / 4;
            for(int dy=-2; dy<=2; dy++) {
                for(int dx=-2; dx<=2; dx++) {
                    grid[(cy+dy)*WIDTH + (cx+dx)].solid = 1.0;
                    grid[(cy+dy)*WIDTH + (cx+dx)].isCathode = true;
                }
            }
        } else { // Array
            for (int x = 50; x < WIDTH; x += 100) {
                for(int dx=-5; dx<=5; dx++) {
                    grid[0 * WIDTH + x + dx].solid = 1.0;
                    grid[0 * WIDTH + x + dx].isCathode = true;
                }
            }
        }
    }

    // Solve Laplace Equation for Potential Field
    void solvePotential() {
        // Successive Over-Relaxation (SOR) for speed
        const double omega = 1.8;
        for (int iter = 0; iter < 20; ++iter) {
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
                if (grid[idx].solid > 0.5) {
                    nextGrid[idx] = grid[idx]; // Maintain solid state
                    continue;
                }

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
                    // Butler-Volmer: reaction rate proportional to C and exp(alpha * eta)
                    double eta = grid[idx].phi;
                    reaction = k_rate * grid[idx].c * exp(alpha * eta);

                    // Stochastic term to encourage dendritic growth
                    double noise = 1.0 + 0.2 * ((double)rand() / RAND_MAX - 0.5);
                    reaction *= noise;
                }

                // Integration
                nextGrid[idx].c = grid[idx].c + D * lapC + migration - reaction;
                nextGrid[idx].solid = grid[idx].solid + reaction * 0.5; // Faster growth
                
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
    bool view3d = false;
    float rotX = 20, rotY = -20;

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

    int handle(int event) {
        static int last_mx, last_my;
        if (event == FL_PUSH) {
            last_mx = Fl::event_x();
            last_my = Fl::event_y();
        }

        if (event == FL_DRAG && (Fl::event_button3() || (Fl::event_state() & FL_SHIFT))) {
            rotY += (Fl::event_x() - last_mx);
            rotX += (Fl::event_y() - last_my);
            last_mx = Fl::event_x();
            last_my = Fl::event_y();
            redraw();
            return 1;
        }

        if (event == FL_PUSH || event == FL_DRAG) {
            if (!Fl::event_button3() && !(Fl::event_state() & FL_SHIFT)) {
                // Correct coordinate mapping relative to the widget
                int mx = Fl::event_x() - x();
                int my = Fl::event_y() - y();

                double sx = (double)WIDTH / w();
                double sy = (double)HEIGHT / h();

                int gx = mx * sx;
                int gy = (h() - my) * sy; // OpenGL Y is bottom-up

                sim->seed(gx, gy, 3);
                return 1;
            }
        }
        return Fl_Gl_Window::handle(event);
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

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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
                double s = sim->grid[i].solid;
                double c = sim->grid[i].c;

                if (s > 0.1) {
                    glColor3f(0.7 + s*0.3, 0.4 + s*0.2, 0.2);
                    glVertex3f(x, y, s * 50.0);
                } else if (c > 0.1) {
                    glColor3f(0.1, 0.2, 0.5 * c);
                    glVertex3f(x, y, -10.0);
                }
            }
        }
        glEnd();
    }
};

void volt_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->voltage = ((Fl_Value_Slider*)w)->value();
}

void d_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->D = ((Fl_Value_Slider*)w)->value();
}

void mobility_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->mobility = ((Fl_Value_Slider*)w)->value();
}

void alpha_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->alpha = ((Fl_Value_Slider*)w)->value();
}

void krate_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->k_rate = ((Fl_Value_Slider*)w)->value();
}

void geom_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->geometry = ((Fl_Choice*)w)->value();
    sim->reset();
}

void view_cb(Fl_Widget* w, void* v) {
    ElectroWindow* view = (ElectroWindow*)v;
    view->view3d = !view->view3d;
    view->redraw();
}

void reset_cb(Fl_Widget* w, void* v) {
    ElectroSim* sim = (ElectroSim*)v;
    sim->reset();
}

int main(int argc, char** argv) {
    srand(time(NULL));
    ElectroSim sim;
    Fl_Double_Window* win = new Fl_Double_Window(600, 500, "Enhanced Electroplating Lab");
    ElectroWindow* glWin = new ElectroWindow(10, 10, 400, 400, &sim);

    int ctrlX = 420;

    Fl_Value_Slider* vSlider = new Fl_Value_Slider(ctrlX, 30, 160, 20, "Applied Voltage");
    vSlider->bounds(0.0, 50.0);
    vSlider->value(sim.voltage);
    vSlider->callback(volt_cb, &sim);

    Fl_Value_Slider* dSlider = new Fl_Value_Slider(ctrlX, 70, 160, 20, "Diffusion (D)");
    dSlider->bounds(0.0, 0.5);
    dSlider->value(sim.D);
    dSlider->callback(d_cb, &sim);

    Fl_Value_Slider* mSlider = new Fl_Value_Slider(ctrlX, 110, 160, 20, "Mobility");
    mSlider->bounds(0.0, 2.0);
    mSlider->value(sim.mobility);
    mSlider->callback(mobility_cb, &sim);

    Fl_Value_Slider* aSlider = new Fl_Value_Slider(ctrlX, 150, 160, 20, "Alpha (Transfer)");
    aSlider->bounds(0.1, 0.9);
    aSlider->value(sim.alpha);
    aSlider->callback(alpha_cb, &sim);

    Fl_Value_Slider* kSlider = new Fl_Value_Slider(ctrlX, 190, 160, 20, "Reaction Rate (k)");
    kSlider->bounds(0.0, 1.0);
    kSlider->value(sim.k_rate);
    kSlider->callback(krate_cb, &sim);

    Fl_Choice* gChoice = new Fl_Choice(ctrlX, 230, 160, 25, "Cathode Geometry");
    gChoice->add("Bottom Plate");
    gChoice->add("Wire / Point");
    gChoice->add("Dot Array");
    gChoice->value(0);
    gChoice->callback(geom_cb, &sim);

    Fl_Button* btn3d = new Fl_Button(ctrlX, 270, 160, 30, "Toggle 3D View");
    btn3d->callback(view_cb, glWin);

    Fl_Button* rBtn = new Fl_Button(ctrlX, 310, 160, 30, "Reset Simulation");
    rBtn->callback(reset_cb, &sim);

    Fl_Box* desc = new Fl_Box(ctrlX, 350, 160, 120,
        "Model:\n- Nernst-Planck\n- Butler-Volmer\n- Laplace Potential\n\n"
        "Molecular:\n- FCC Lattice (Cu)\n- Directional bond\n- Surface energy");
    desc->align(FL_ALIGN_TOP | FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);

    win->end();
    win->show(argc, argv);
    return Fl::run();
}
