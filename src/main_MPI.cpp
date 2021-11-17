#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body_MPI.hpp>
#include <chrono>

#define MIN_BODY_PER_PROC 10
#define DEBUGx
#define TEST_ITERATION 1

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: " << argv[0] << " <THREAD_SIZE> <GRAPH>" << std::endl;
        return 0;
    }
    static int bodies = std::atoi(argv[1]); // number of the bodies, fixed
    static int GUI = std::atoi(argv[2]);

    if (bodies <= 0)
    {
        std::cerr << "BODY should be greater than 0" << std::endl;
        return 0;
    }

    size_t duration = 0;
    int iteration = 0;

    int rank, wsize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

#ifdef DEBUG
    std::cout << "rank: " << rank << "; bodies: " << bodies << std::endl;
#endif

    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;

    static float elapse = 1.0;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;

    BodyPool pool(static_cast<size_t>(bodies), space, max_mass); // the main body pool

    // each proc calculate their global offset and data size
    int *displs, *scounts, offset = 0, cur_rank = 0, worker_proc = 0, remain = 0;
    displs = (int *)malloc(wsize * sizeof(int));  // workload record [global offset] for each proc
    scounts = (int *)malloc(wsize * sizeof(int)); // workload record [local size] for each proc

    // calculate individual workload
    // [1] Each proc get jobs to do
    if (bodies >= wsize * MIN_BODY_PER_PROC)
    {
        offset = 0;
        for (cur_rank = 0; cur_rank < wsize; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = std::ceil(((float)bodies - cur_rank) / wsize);
            offset += scounts[cur_rank];
        }
    }
    // [2] Some proc has no job to do
    else
    {
        offset = 0;
        worker_proc = std::ceil((float)bodies / MIN_BODY_PER_PROC);
        remain = bodies % MIN_BODY_PER_PROC;
        for (cur_rank = 0; cur_rank < worker_proc - 1; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = MIN_BODY_PER_PROC;
            offset += MIN_BODY_PER_PROC;
        }
        displs[cur_rank] = offset;
        scounts[cur_rank] = remain == 0 ? MIN_BODY_PER_PROC : remain;
        offset += scounts[cur_rank];
        cur_rank++;
        for (; cur_rank < wsize; ++cur_rank)
        {
            displs[cur_rank] = offset;
            scounts[cur_rank] = 0;
        }
    }
    int localoffset = displs[rank];
    int localsize = scounts[rank];

    // validate the workload distribution result
#ifdef DEBUG
    std::cout << "[rank] " << rank << " [localoffset] " << localoffset << " [localsize] " << localsize << std::endl;
#endif

    if (GUI == 1)
    {
        if (rank == 0)
        {
            graphic::GraphicContext context{"Assignment 3"};
            context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                        {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 3", nullptr,
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

        if (iteration == 600000){
            exit(0);
        }

        auto begin = std::chrono::high_resolution_clock::now();
        // send all the data to all the proc from ROOT
        MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);   
        pool.copy_x = pool.x;
        pool.copy_y = pool.y;


        
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();

            // begin to update the position on its portion [ROOT]
            pool.update_for_tick(elapse, gravity, space, radius, localoffset, localsize);
            
            // get results from other proc
            MPI_Gatherv(pool.copy_x.data() + localoffset, localsize, MPI_DOUBLE, pool.x.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.copy_y.data() + localoffset, localsize, MPI_DOUBLE, pool.y.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.copy_vx.data() + localoffset, localsize, MPI_DOUBLE, pool.vx.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.copy_vy.data() + localoffset, localsize, MPI_DOUBLE, pool.vy.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.ax.data() + localoffset, localsize, MPI_DOUBLE, pool.ax.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pool.ay.data() + localoffset, localsize, MPI_DOUBLE, pool.ay.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();
            duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();

            std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
            auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
            std::cout << "speed: " << speed << " bodies per second" << std::endl;
            duration = 0;
            ++iteration;
            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End(); });
        }

        // the other rank continuously calculate the results and send back
        else if (rank != 0)
        {
            while (1)
            {
                if (iteration == 600000)
                {
                    exit(0);
                }
                // receive all the data from ROOT
                MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                pool.copy_x = pool.x;
                pool.copy_y = pool.y;


                // calculate the reposible portion [Other not ROOT]
                pool.update_for_tick(elapse, gravity, space, radius, localoffset, localsize);

                // send results back to ROOT
                MPI_Gatherv(pool.copy_x.data() + localoffset, localsize, MPI_DOUBLE, pool.x.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_y.data() + localoffset, localsize, MPI_DOUBLE, pool.y.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vx.data() + localoffset, localsize, MPI_DOUBLE, pool.vx.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vy.data() + localoffset, localsize, MPI_DOUBLE, pool.vy.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ax.data() + localoffset, localsize, MPI_DOUBLE, pool.ax.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ay.data() + localoffset, localsize, MPI_DOUBLE, pool.ay.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                ++iteration;
            }
        }
        MPI_Finalize();
        return 0;
    }

    else if (GUI == 0)
    {
        if (rank == 0)
        {
            while (1)
            {
                if (iteration == TEST_ITERATION)
                {
                    MPI_Finalize();
                    exit(0);
                }
                auto begin = std::chrono::high_resolution_clock::now();
                // send all the data to all the proc from ROOT
                MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                pool.copy_x = pool.x;
                pool.copy_y = pool.y;

                // begin to update the position on its portion [ROOT]
                pool.update_for_tick(elapse, gravity, space, radius, localoffset, localsize);
                
                // get results from other proc
                MPI_Gatherv(pool.copy_x.data() + localoffset, localsize, MPI_DOUBLE, pool.x.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_y.data() + localoffset, localsize, MPI_DOUBLE, pool.y.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vx.data() + localoffset, localsize, MPI_DOUBLE, pool.vx.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vy.data() + localoffset, localsize, MPI_DOUBLE, pool.vy.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ax.data() + localoffset, localsize, MPI_DOUBLE, pool.ax.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ay.data() + localoffset, localsize, MPI_DOUBLE, pool.ay.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                auto end = std::chrono::high_resolution_clock::now();
                duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();

                std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
                auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
                std::cout << "speed: " << speed << " bodies per second" << std::endl;
                duration = 0;
                ++iteration;

            }
        }

        // the other rank continuously calculate the results and send back
        else if (rank != 0)
        {
            while (1)
            {
                if (iteration == TEST_ITERATION)
                {
                    MPI_Finalize();
                    exit(0);
                }
                // receive all the data from ROOT
                MPI_Bcast(pool.x.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.y.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vx.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.vy.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ax.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.ay.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(pool.m.data(), bodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                pool.copy_x = pool.x;
                pool.copy_y = pool.y;
#ifdef DEBUG
                for (int j = 0; j < bodies; j++)
                    assert(pool.x[j] == pool.copy_x[j]);
#endif

                // calculate the reposible portion [Other not ROOT]
                pool.update_for_tick(elapse, gravity, space, radius, localoffset, localsize);

                // send results back to ROOT
                MPI_Gatherv(pool.copy_x.data() + localoffset, localsize, MPI_DOUBLE, pool.x.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_y.data() + localoffset, localsize, MPI_DOUBLE, pool.y.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vx.data() + localoffset, localsize, MPI_DOUBLE, pool.vx.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.copy_vy.data() + localoffset, localsize, MPI_DOUBLE, pool.vy.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ax.data() + localoffset, localsize, MPI_DOUBLE, pool.ax.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(pool.ay.data() + localoffset, localsize, MPI_DOUBLE, pool.ay.data(), scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                ++iteration;
            }
        }
        MPI_Finalize();
        return 0;
    }
}
