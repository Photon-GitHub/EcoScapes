import json
import os
import tarfile
from datetime import timedelta, date
from typing import override

from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from sentinelhub import (
    BBox,
    CRS,
    SHConfig,
    DataCollection,
    MimeType,
    SentinelHubRequest,
)

from modules.module import Module, ModuleResult

country: str = "Germany"


def load_credentials(file_path):
    """Load credentials from a JSON file.
    :param file_path: Path to the credentials JSON file.
    :return: Dictionary with credentials.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def generate_new_credentials_config():
    """Generate a new Sentinel Hub configuration using credentials from the secrets file."""
    print("Generating new EcoScapes config.")
    config = SHConfig()
    credentials = load_credentials('../secret/secrets.json')
    config.sh_client_id = credentials['sh_client_id']
    config.sh_client_secret = credentials['sh_client_secret']
    config.save("EcoScapes")
    return config


def setup_credentials(force_load=False):
    """Setup Sentinel Hub configuration using credentials from a file.
    :return: Configured Sentinel Hub configuration object.
    """
    if force_load:
        return generate_new_credentials_config()

    try:
        config = SHConfig("EcoScapes")
        print("Found existing EcoScapes config.")
        return config
    except KeyError:
        return generate_new_credentials_config()


def create_oauth_session(config):
    """Create and return an OAuth2 session using Sentinel Hub credentials.
    :param config: Sentinel Hub configuration object.
    :return: Authenticated OAuth2 session.
    """
    client = BackendApplicationClient(client_id=config.sh_client_id)
    oauth = OAuth2Session(client=client)

    oauth.fetch_token(
        token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
        client_secret=config.sh_client_secret,
        include_client_id=True
    )

    oauth.get("https://services.sentinel-hub.com/configuration/v1/wms/instances")

    def sentinelhub_compliance_hook(response):
        response.raise_for_status()
        return response

    oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)
    return oauth


def time_range_formatted_for_request(days):
    """Calculate start and end dates for the given number of past days.
    :param days: Number of days in the past for the time range.
    :return: Tuple of start and end dates formatted as strings.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_location_bounds(location_name):
    """Get the bounding box for a given location using geopy.
    :param location_name: Name of the location to geocode.
    :return: Bounding box coordinates (north, south, east, west).
    :raises ValueError: If the location or bounding box cannot be found.
    """
    geolocator = Nominatim(user_agent="EcoScapes")
    location = geolocator.geocode(f"{location_name}, {country}")

    if location is None:
        raise ValueError(f"Location not found: {location_name}")

    bbox = location.raw.get('boundingbox', None)
    if bbox is None:
        raise ValueError(f"Bounding box not found for {location_name}")

    location_bounds = (bbox[2], bbox[0], bbox[3], bbox[1])
    return location_bounds


def get_location_bounds_with_radius(location_name, radius_km=5):
    """
    Get the bounding box for a given location using a radius around the city center.
    :param location_name: Name of the location to geocode.
    :param radius_km: Radius in kilometers around the city center to create the bounding box.
    :return: Bounding box coordinates (north, south, east, west).
    :raises ValueError: If the location cannot be found.
    """
    geolocator = Nominatim(user_agent="EcoScapes")
    location = geolocator.geocode(f"{location_name}, {country}")

    if location is None:
        raise ValueError(f"Location not found: {location_name}")

    # City center coordinates
    center_lat = location.latitude
    center_lon = location.longitude

    # Calculate the coordinates of the bounding box
    west = geodesic(kilometers=radius_km).destination((center_lat, center_lon), 270).longitude
    south = geodesic(kilometers=radius_km).destination((center_lat, center_lon), 180).latitude
    east = geodesic(kilometers=radius_km).destination((center_lat, center_lon), 90).longitude
    north = geodesic(kilometers=radius_km).destination((center_lat, center_lon), 0).latitude

    location_bounds = (west, south, east, north)
    return location_bounds


def prepare_satellite_image_request(location_bounds, config, resolution=1024):
    """Prepare a Sentinel Hub request for satellite images of a location.
    :param location_bounds: Bounding box coordinates of the location.
    :param config: Sentinel Hub configuration object.
    :param resolution: The downloaded satellite images have the size of resolution x resolution pixels.
    :return: Prepared Sentinel Hub request object.
    """
    evalscript = """
        //VERSION=3
        const moistureRamps = [
            [-0.8, 0x800000],
            [-0.24, 0xff0000],
            [-0.032, 0xffff00],
            [0.032, 0x00ffff],
            [0.24, 0x0000ff],
            [0.8, 0x000080]
        ];
    
        const viz = new ColorRampVisualizer(moistureRamps);
    
        function setup() {
          return {
            input: ["B02", "B03", "B04", "B08", "B8A", "B11", "dataMask"],
            output: [
              { id: "rgb", bands: 3 },
              { id: "moisture", bands: 4 },
              { id: "water", bands: 1 }
            ]
          };
        }
    
        function evaluatePixel(sample) {
          let moisture = index(sample.B8A, sample.B11);
          let water = index(sample.B03, sample.B08);
          return {
            rgb: [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02],
            moisture: [...viz.process(moisture), sample.dataMask],
            water: [water]
          };
        }
        """

    start_date, end_date = time_range_formatted_for_request(days=360)
    bbox = BBox(bbox=location_bounds, crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                maxcc=1.0
            )
        ],
        responses=[
            SentinelHubRequest.output_response('rgb', MimeType.PNG),
            SentinelHubRequest.output_response('moisture', MimeType.PNG),
            SentinelHubRequest.output_response('water', MimeType.PNG)
        ],
        bbox=bbox,
        size=(resolution, resolution),
        config=config,
        data_folder="./satellite_data",
    )

    return request


def untar_files_in_path(path):
    """Extracts all .tar files in the given directory and removes the original .tar files after extraction.
    :param path: str, the path to the directory containing the .tar files.
    """
    # Check if the provided path exists and is a directory
    if not os.path.isdir(path):
        raise ValueError(f"The provided path '{path}' is not a valid directory.")

    tar_paths = [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".tar")]

    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=path)
        os.remove(tar_path)


class SatelliteLoader(Module):
    def __init__(self):
        super().__init__("SatelliteLoader", {"LocationExtraction"})

    @override
    def main(self) -> ModuleResult:
        """Main function to download satellite images for a specified location."""
        satellite_image_folder_path = os.path.realpath("./satellite_data")

        location_name = self.load_location()
        resolution = 1024

        os.makedirs(satellite_image_folder_path, exist_ok=True)

        location_folder = os.path.join(satellite_image_folder_path, location_name)

        if os.path.exists(location_folder):
            print(f"Data for {location_name} already exists in {location_folder}. Skipping download.")
            return ModuleResult.OK

        config = setup_credentials(force_load=False)
        create_oauth_session(config)

        location_bounds = get_location_bounds_with_radius(location_name, radius_km=3)
        print(location_bounds)

        dir_before = set(os.listdir(satellite_image_folder_path))

        request = prepare_satellite_image_request(location_bounds, config, resolution=resolution)
        request.get_data(save_data=True)
        print(f'Downloaded images for {location_name}.')

        dir_after = set(os.listdir(satellite_image_folder_path))
        new_dirs = dir_after - dir_before

        for dir_name in new_dirs:
            old_dir_path = os.path.join(satellite_image_folder_path, dir_name)
            new_dir_path = os.path.join(satellite_image_folder_path, location_name)
            os.rename(old_dir_path, new_dir_path)

            untar_files_in_path(new_dir_path)
            print(f'Renamed {old_dir_path} to {new_dir_path}')

        return ModuleResult.OK
