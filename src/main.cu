#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body_CUDA.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREAD_PER_BLOCK 1024

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

__device__
void mutual_check_and_update(double *x, double *y, double * vx, double * vy, double *ax, double *ay, double *m,
    double *x_buf, double * y_buf, double * vx_buf, double * vy_buf, 
    int i, int j, double radius, double gravity, double COLLISION_RATIO)
{
    double delta_x = x[i] - x[j];
    double delta_y = y[i] - y[j];
    double distance_square = delta_x * delta_x + delta_y * delta_y;

    double ratio = 1 + COLLISION_RATIO;
    if (distance_square < radius * radius)
    {
        distance_square = radius * radius;
    }
    double distance = std::sqrt(distance_square);
    if (distance < radius)
    {
        distance = radius;
    }
    if (distance_square <= radius * radius)
    {
        double dot_prod = delta_x * (vx[i] - vx[j]) + delta_y * (vy[i] - vy[j]);
        double scalar = 2 / (m[i] + m[j]) * dot_prod / distance_square;
        vx_buf[i] -= scalar * delta_x * m[j];
        vy_buf[j] -= scalar * delta_y * m[j];

        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        x_buf[i] += delta_x / distance * ratio * radius / 2.0;
        y_buf[i] += delta_y / distance * ratio * radius / 2.0;
    }
    else
    {
        // update acceleration only when no collision
        auto scalar = gravity / distance_square / distance;
        ax[i] -= scalar * delta_x * m[j];
        ay[i] -= scalar * delta_y * m[j];
    }
}

__device__
void handle_wall_collision(double *ax, double *ay, double *x_buf, double *y_buf, double *vx_buf, double *vy_buf, 
                           int i, double position_range, double radius, double COLLISION_RATIO) {
    bool flag = false;
    if (x_buf[i]<= radius)
    {
        flag = true;
        x_buf[i]= radius + radius * COLLISION_RATIO;
        vx_buf[i]= -vx_buf[i];
    }
    else if (x_buf[i] >= position_range - radius)
    {
        flag = true;
        x_buf[i] = position_range - radius - radius * COLLISION_RATIO;
        vx_buf[i] = -vx_buf[i];
    }

    if (y_buf[i] <= radius)
    {
        flag = true;
        y_buf[i] = radius + radius * COLLISION_RATIO;
        vy_buf[i] = -vy_buf[i];
    }
    else if (y_buf[i] >= position_range - radius)
    {
        flag = true;
        y_buf[i] = position_range - radius - radius * COLLISION_RATIO;
        vy_buf[i] = -vy_buf[i];
    }
    if (flag)
    {
        ax[i] = 0;
        ay[i] = 0;
    }
}

__device__
void update_single(double *x_buf, double *y_buf, double *ax, double *ay, double *vx_buf, double *vy_buf, int i, double elapse, double position_range, double radius, double COLLISION_RATIO)
{
    vx_buf[i] += ax[i] * elapse;
    vy_buf[i] += ay[i] * elapse;
    handle_wall_collision(ax, ay, x_buf, y_buf, vx_buf, vy_buf, i, position_range, radius, COLLISION_RATIO);
    x_buf[i] += vx_buf[i] * elapse;
    y_buf[i] += vy_buf[i] * elapse;
    handle_wall_collision(ax, ay, x_buf, y_buf, vx_buf, vy_buf, i, position_range, radius, COLLISION_RATIO); // change x & v in buffer "copy"
}

__global__ 
void update_for_all(
    double *x_buf,
    double *y_buf,
    double *vx_buf,
    double *vy_buf,
    int COLLISION_RATIO,
    int bodies,
    int * displs,
    int * scounts,
    double * x,
    double * y,
    double * vx,
    double * vy,
    double * ax,
    double * ay,
    double * m,
    double elapse,
    double gravity,
    double position_range,
    double radius) 
{
    int localoffset = displs[threadIdx.x]; //tmp variable
    int localsize = scounts[threadIdx.x];  //tmp variable
    // zero accleration
    for (int i = 0; i < bodies; ++i) {
        ax[i] = 0;
        ay[i] = 0;
    }
    for (size_t i = localoffset; i < localoffset + localsize; ++i)
    {
        for (size_t j = 0; j < bodies; ++j)
            mutual_check_and_update(x, y, vx, vy, ax, ay, m, x_buf, y_buf, vx_buf, vy_buf, i, j, radius, gravity, COLLISION_RATIO);
    }
    __syncthreads();
    for (size_t i = localoffset; i < localoffset + localsize; ++i)
        update_single(x_buf, y_buf, ax, ay, vx_buf, vy_buf, i, elapse, position_range, radius, COLLISION_RATIO);
    for (size_t i = localoffset; i < localoffset + localsize; ++i) {
        x[i] = x_buf[i];
        y[i] = y_buf[i];
        vx[i] = vx_buf[i];
        vy[i] = vy_buf[i];
    }
}


int main(int argc, char **argv) {
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <BODIES>" << std::endl;
        return 0;
    }
    static int bodies = std::atoi(argv[1]);
    if (bodies <= 0)
    {
        std::cerr << "BODY should be greater than 0" << std::endl;
        return 0;
    }

    int THREAD = bodies <= MAX_THREAD_PER_BLOCK? bodies:MAX_THREAD_PER_BLOCK; // total number of threads using in the program


    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;

    int *displs, *scounts, i, offset = 0;
    displs = (int *)malloc(THREAD * sizeof(int));
    scounts = (int *)malloc(THREAD * sizeof(int));

    // calculate individual workload
    if (THREAD == bodies) {
        for (i = 0; i < THREAD; ++i) {
            displs[i] = offset;
            scounts[i] = 1;
            ++offset;
        }
    }
    else if (THREAD < bodies) {
        for (i = 0; i < THREAD; ++i) {
            displs[i] = offset;
            scounts[i] = std::ceil(((float)bodies - i) / THREAD);
            offset += scounts[i];
        }
    }
    
    // body pool initialization
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    double coll_ratio = pool.COLLISION_RATIO;

    // copies in device
    double cuda_elapse;
    double cuda_gravity;
    double cuda_position_range;
    double cuda_radius;
    double* cuda_x, *cuda_y, *cuda_vx, *cuda_vy, *cuda_ax, *cuda_ay, *cuda_m, cuda_coll_ratio, *cuda_x_buf,
          * cuda_y_buf, *cuda_vx_buf, *cuda_vy_buf;
    int *cuda_displs, *cuda_scounts, cuda_bodies;

    // copy displs, scounts to device
    cudaMalloc((void **)&cuda_displs, sizeof(int) * THREAD);
    cudaMemcpy(cuda_displs, displs, sizeof(int) * THREAD, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&cuda_scounts, sizeof(int) * THREAD);
    cudaMemcpy(cuda_scounts, scounts, sizeof(int) * THREAD, cudaMemcpyHostToDevice);

    // allocate space for device copies of data in pool for CALCULATION
    cudaMalloc((void **)&cuda_bodies, sizeof(int));
    cudaMalloc((void **)&cuda_elapse, sizeof(double));
    cudaMalloc((void **)&cuda_gravity, sizeof(double));
    cudaMalloc((void **)&cuda_position_range, sizeof(double));
    cudaMalloc((void **)&cuda_radius, sizeof(double));
    cudaMalloc((void **)&cuda_coll_ratio, sizeof(double));
    cudaMalloc((void **)&cuda_x, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_y, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vx, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vy, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_ax, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_ay, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_m, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_x_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_y_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vx_buf, sizeof(double) * bodies);
    cudaMalloc((void **)&cuda_vy_buf, sizeof(double) * bodies);

    // copy data to device
    cudaMemcpy((void **)&cuda_bodies, &bodies, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&cuda_elapse, &elapse, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&cuda_gravity, &gravity, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&cuda_position_range, &space, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&cuda_radius, &radius, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void **)&cuda_coll_ratio, &coll_ratio, sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_x, pool.x.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_x_buf, pool.x.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_vx, pool.vx.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vx_buf, pool.vx.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_y, pool.y.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y_buf, pool.y.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_vy, pool.vy.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_vy_buf, pool.vy.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);
   
    cudaMemcpy(cuda_m, pool.m.data(), sizeof(double) * bodies, cudaMemcpyHostToDevice);

    graphic::GraphicContext context{"Assignment 3"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::ColorEdit4("Color", &color.x);


        update_for_all<<<1, THREAD>>>
        (
            cuda_x_buf,
            cuda_y_buf,
            cuda_vx_buf,
            cuda_vy_buf,
            cuda_coll_ratio,
            cuda_bodies,
            cuda_displs,
            cuda_scounts,
            cuda_x,
            cuda_y,
            cuda_vx,
            cuda_vy,
            cuda_ax,
            cuda_ay,
            cuda_m,
            cuda_elapse,
            cuda_gravity,
            cuda_position_range,
            cuda_radius);

        // collect position result for GUI display
        cudaMemcpy(pool.x.data(), cuda_x, sizeof(double)*bodies, cudaMemcpyDeviceToHost);
        cudaMemcpy(pool.x.data(), cuda_x, sizeof(double)*bodies, cudaMemcpyDeviceToHost);

        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            // pool.update_for_tick(elapse, gravity, space, radius);
            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End();
    });
}
