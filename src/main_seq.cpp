#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <chrono>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

int main(int argc, char **argv)
{
    size_t duration = 0;
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 1.0;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

        {
        graphic::GraphicContext context{"Assignment 2"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                    {
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
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);
        if (current_space != space || current_bodies != bodies || current_max_mass != max_mass) {
            space = current_space;
            bodies = current_bodies;
            max_mass = current_max_mass;
            pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
        }
        std::cout<<bodies<<std::endl;
        {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            auto begin = std::chrono::high_resolution_clock::now();
            pool.update_for_tick(elapse, gravity, space, radius);
            auto end = std::chrono::high_resolution_clock::now();
            duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();

            std::cout << bodies << " bodies in last " << duration << " nanoseconds\n";
            auto speed = static_cast<double>(bodies) / static_cast<double>(duration) * 1e9;
            std::cout << "speed: " << speed << " bodies per second" << std::endl;
            duration = 0;

            for (size_t i = 0; i < pool.size(); ++i) {
                auto body = pool.get_body(i);
                auto x = p.x + static_cast<float>(body.get_x());
                auto y = p.y + static_cast<float>(body.get_y());
                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
            }
        }
        ImGui::End(); });
    }
}
