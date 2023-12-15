flake8 --max-line-length=120 \
       --exclude WeatherRoutingTool/additional_code \
       --extend-ignore F401,F403,F405,E711 ./WeatherRoutingTool ./execute_routing.py ./compare_routes.py ./tests
