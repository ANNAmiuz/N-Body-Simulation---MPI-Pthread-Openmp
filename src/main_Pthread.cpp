#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body_Pthread.hpp>
#include <chrono>

#define DEBUGx
#define TEST_ITERATION 1

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: " << argv[0] << " <BODIES> <THREAD_SIZE> <GRAPH>" << std::endl;
        return 0;
    }
    static int bodies = std::atoi(argv[1]);
    static int THREAD = std::atoi(argv[2]);
    int GUI = std::atoi(argv[3]);
    static int iteration = 0;

    if (THREAD <= 0 || bodies <= 0)
    {
        std::cerr << "BODY & THREAD_SIZE should be greater than 0" << std::endl;
        return 0;
    }
    size_t duration = 0;
    std::vector<std::thread> thread_pool{};

    int i, offset = 0;
    size_t *startidx, *localsize;
    startidx = (size_t *)malloc(THREAD * sizeof(size_t));
    localsize = (size_t *)malloc(THREAD * sizeof(size_t));
    for (i = 0; i < THREAD; ++i)
    {
        startidx[i] = offset;
        localsize[i] = std::ceil(((float)bodies - i) / THREAD);
        offset += localsize[i];
    }
#ifdef DEBUG
    for (i = 0; i < THREAD; ++i)
    {
        std::cout << "start: " << startidx[i] << "local: " << localsize[i] << std::endl;
    }
#endif

    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static float elapse = 0.91;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    pool.copy_x = pool.x;
    pool.copy_y = pool.y;

    if (GUI == 1)
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

        auto begin = std::chrono::high_resolution_clock::now();
        if (iteration == 600000)
        {
            exit(0);
        }
        // initialize some threads to do the job
        for (i = 0; i < THREAD; ++i) 
        {
            thread_pool.push_back(std::thread(
                [&] { pool.update_for_tick(elapse, gravity, space, radius, startidx[i], localsize[i]); }));
        }

        // join all the threads
        for (i = 0; i < THREAD; ++i) 
            thread_pool[i].join();

        auto end = std::chrono::high_resolution_clock::now();
        duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
        
        // clear the used threads
        thread_pool.clear();
        pool.x = pool.copy_x;
        pool.y = pool.copy_y;
        pool.vx = pool.copy_vx;
        pool.vy = pool.copy_vy;
        

        std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
        auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
        std::cout << "speed: " << speed << " bodies per second" << std::endl;
        duration = 0;
        ++iteration;


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
        ImGui::End(); });
    }
    else if (GUI == 0)
    {
        while (1)
        {
            if (iteration == TEST_ITERATION)
            {
                exit(0);
            }
            auto begin = std::chrono::high_resolution_clock::now();
            // initialize some threads to do the job
            for (i = 0; i < THREAD; ++i)
            {
                thread_pool.push_back(std::thread(
                    [&]
                    { pool.update_for_tick(elapse, gravity, space, radius, startidx[i], localsize[i]); }));
            }

            // join all the threads
            for (i = 0; i < THREAD; ++i)
                thread_pool[i].join();

            auto end = std::chrono::high_resolution_clock::now();
            duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();

            // clear the used threads
            thread_pool.clear();
            pool.x = pool.copy_x;
            pool.y = pool.copy_y;
            pool.vx = pool.copy_vx;
            pool.vy = pool.copy_vy;

            std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
            auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
            std::cout << "speed: " << speed << " bodies per second" << std::endl;
            duration = 0;
            ++iteration;
        }
    }
}
