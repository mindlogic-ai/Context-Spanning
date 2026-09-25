# Tool bank used by this package

DuetaSpan's bank has 125 tools (BFCL/SGD-derived schemas, real execution). This package ships **60 of them**.
The criterion: a tool belongs in a spoken conversation only if its result is something the assistant would say
to the listener — a current value, a place, a route, a booking, a search result. Tools whose output is a web page,
a file, a coordinate pair, an app deep-link or a merchant back-office record are not spoken and are left out.
In addition to the bank, the four MCP servers (time, weather, finance, web search) are always in the router's
universe, so the router sees 64 tools (63 on a machine without a `map_road_traffic` API key).

## Excluded (65)

- **browser automation (page navigation, clicks, HTTP verbs, test codegen)** (32): `start_codegen_session`, `end_codegen_session`, `get_codegen_session`, `clear_codegen_session`, `playwright_navigate`, `playwright_screenshot`, `playwright_click`, `playwright_iframe_click`, `playwright_iframe_fill`, `playwright_fill`, `playwright_select`, `playwright_hover`, `playwright_evaluate`, `playwright_console_logs`, `playwright_get`, `playwright_post`, `playwright_put`, `playwright_patch`, `playwright_delete`, `playwright_expect_response`, `playwright_assert_response`, `playwright_custom_user_agent`, `playwright_drag`, `playwright_press_key`, `playwright_click_and_switch_tab`, `puppeteer_navigate`, `puppeteer_screenshot`, `puppeteer_click`, `puppeteer_fill`, `puppeteer_select`, `puppeteer_hover`, `puppeteer_evaluate`
- **filesystem (read/write/list files)** (11): `read_file`, `read_multiple_files`, `write_file`, `edit_file`, `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`, `search_files`, `get_file_info`, `list_allowed_directories`
- **web crawling / site mapping (not a question a listener asks)** (3): `tavily-extract`, `tavily-crawl`, `tavily-map`
- **PayPal merchant back-office (invoices, catalog products, subscription plans, shipments, orders, refunds)** (6): `create_invoice`, `create_product`, `create_subscription_plan`, `create_shipment_tracking`, `create_order`, `create_refund`
- **coordinate conversion, IP geolocation, elevation (values nobody says aloud)** (9): `maps_geocode`, `maps_reverse_geocode`, `map_geocode`, `map_reverse_geocode`, `maps_geo`, `maps_regeocode`, `map_ip_location`, `maps_ip_location`, `maps_elevation`
- **app deep-link / share-URL generators (a URI, not speech)** (4): `map_mark`, `maps_schema_personal_map`, `maps_schema_navi`, `maps_schema_take_taxi`

## Included (60)

- map (19): `maps_search_places`, `maps_place_details`, `maps_distance_matrix`, `maps_directions`, `map_search_places`, `map_place_details`, `map_directions_matrix`, `map_directions`, `map_weather`, `map_road_traffic`, `maps_direction_bicycling`, `maps_direction_driving`, `maps_direction_transit_integrated`, `maps_direction_walking`, `maps_distance`, `maps_around_search`, `maps_search_detail`, `maps_text_search`, `maps_weather`
- Movies (4): `Movies_3_FindMovies`, `Movies_1_BuyMovieTickets`, `Movies_1_FindMovies`, `Movies_1_GetTimesForMovie`
- Services (4): `Services_4_BookAppointment`, `Services_4_FindProvider`, `Services_1_BookAppointment`, `Services_1_FindProvider`
- Hotels (4): `Hotels_2_BookHouse`, `Hotels_2_SearchHouse`, `Hotels_4_ReserveHotel`, `Hotels_4_SearchHotel`
- Flights (2): `Flights_4_SearchOnewayFlight`, `Flights_4_SearchRoundtripFlights`
- RentalCars (2): `RentalCars_3_GetCarsAvailable`, `RentalCars_3_ReserveCar`
- Restaurants (2): `Restaurants_2_ReserveRestaurant`, `Restaurants_2_FindRestaurants`
- Media (2): `Media_3_FindMovies`, `Media_3_PlayMovie`
- Music (2): `Music_3_PlayMedia`, `Music_3_LookupMusic`
- Alarm (2): `Alarm_1_GetAlarms`, `Alarm_1_AddAlarm`
- Events (2): `Events_3_FindEvents`, `Events_3_BuyEventTickets`
- Buses (2): `Buses_3_FindBus`, `Buses_3_BuyBusTicket`
- Trains (2): `Trains_1_GetTrainTickets`, `Trains_1_FindTrains`
- Payment (2): `Payment_1_RequestPayment`, `Payment_1_MakePayment`
- Homes (2): `Homes_2_FindHomeByArea`, `Homes_2_ScheduleVisit`
- search (2): `tavily-search`, `search`
- Weather (1): `Weather_1_GetWeather`
- RideSharing (1): `RideSharing_2_GetRide`
- Travel (1): `Travel_1_FindAttractions`
- Messaging (1): `Messaging_1_ShareLocation`
- finance (1): `get_stock_price_global_market`

`contextspan/datasets/moshicp/mcp_tool_bank.json` is this curated bank.
