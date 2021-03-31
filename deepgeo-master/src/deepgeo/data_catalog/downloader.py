import deepgeo.data_catalog.espa_downloader as ed

downloader = ed.EspaDownloader()
downloader.authenticate(username='micheldearaujo', password='86558179Khan')

downloader.get_intersections('prodes_shp_crop.shp')

downloader.plot_intersections()

bulk, ids, notfound = downloader.consult_dates(start_date='2018-01-01',
                                               end_date='2018-12-31',
                                               max_cloud_cover=20)

downloader.get_available_products()
downloader.get_available_projections()

downloader.generate_order(products=['sr','ndvi'],
                          file_format='gtiff',
                          projection='lonlat',
                          verbose=True)

downloader.place_order()

orders=downloader.list_orders()

if downloader.is_order_complete(orders[0]):
    downloader.download_order(orders[0], output_dir='./images')