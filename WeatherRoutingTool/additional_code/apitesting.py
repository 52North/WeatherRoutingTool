import cdstoolbox as ct

@ct.application(title='Download data')
@ct.output.download()
def download_application():
    data = ct.catalogue.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': '10',
            'year': '2022',
            'month': '06',
            'day': '14',
            'time': '17:00',
        }
    )
    return data