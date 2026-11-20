#define main unused_aim_main
#include "aim-live.cpp"
#undef main
#include <cstdlib>
static Detection ball(double x,double y) { return {x-10,y-10,x+10,y+10,.9f}; }
static void require(bool ok,int code) { if(!ok) std::exit(code); }
int main(int argc,char** argv) {
    if(argc!=2) return 64;
    const cv::Size size(1920,1080);
    const auto t=Clock::now();
    Controller c({.9697,.9697,120});
    const std::vector<Detection> scene={ball(1100,600),ball(700,400),ball(1300,800)};
    auto at=[&](int ms,const auto& targets) { auto cap=t+std::chrono::milliseconds(ms); return c.update(targets,size,cap,cap+8ms); };
    auto first=at(0,scene);
    require(first.has_value() && c.lock_generation()==1,1);
    c.sent(t+10ms,*first);
    require(!at(115,scene) && c.feedback()=="waiting",2);
    auto retry=at(131,scene);
    require(retry.has_value() && c.feedback()=="feedback_timeout" && c.lock_generation()==2,3);
    require(std::abs(retry->x)>8 || std::abs(retry->y)>8,4); // normal calibrated correction, not an artificial 8-count recovery cap
    FILE* f=std::fopen(argv[1],"wb"); if(!f)return 5; std::fclose(f);
    { HidMouse mouse(argv[1]); auto result=mouse.move(*retry,Clock::now()+100ms); require(result.reports>0 && result.moved==*retry,6); }
    std::ifstream in(argv[1],std::ios::binary); std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)),{});
    require(!data.empty() && data.size()%3==0,7);
    for(std::size_t i=0;i<data.size();i+=3) require(data[i]==0,8);
    std::cout << "120 ms recovery passed: hold before timeout; then normal calibrated correction is emitted, no extra movement cap and no buttons pressed.\n";
}
