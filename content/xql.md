Title: Sql to excel
Date: 2015-02-02
Slug: xql
Category: Data Posts
Tags: sql, python
Authors: Thomas Buhrmann

A little python tool to execute an sql script (postgresql in this case, but should be easily modifiable for mysql etc.) and store the result in a csv or excel (xls file):

    :::python    
    """
    Executes an sql script and stores the result in a file.
    """
     
    import os, sys
    import subprocess
    import csv
    from xlwt import Workbook
     
     
    def sql_to_csv(sql_fnm, csv_fnm):
        """ Write result of executing sql script to txt file"""
     
        with open(sql_fnm, 'r') as sql_file:
            query = sql_file.read()
            query = "COPY (" + query + ") TO STDOUT WITH CSV HEADER"
            cmd = 'psql -c "' + query + '"'
            print cmd
     
            data = subprocess.check_output(cmd, shell=True)
     
            with open(csv_fnm, 'wb') as csv_file:
                csv_writer = csv.writer(csv_file)
                rows = data.splitlines()
                for row in rows:
                    csv_writer.writerow(row.split(','))
     
     
    def sql_to_xsl(sql_fnm, xls_fnm):
        """ Write result of executing sql script to xls file"""
     
        with open(sql_fnm, 'r') as sql_file:
            query = sql_file.read()
            query = "COPY (" + query + ") TO STDOUT WITH CSV HEADER"
            cmd = 'psql -c "' + query + '"'
            print cmd
     
            data = subprocess.check_output(cmd, shell=True)
     
            book = Workbook()
            sheet = book.add_sheet('Sheet 1')
            rows = data.splitlines()
            for row_idx, row in enumerate(rows):
                values = row.split(',')
                for col_idx, val in enumerate(values):
                    sheet.write(row_idx, col_idx, val)
            book.save(xls_fnm)
     
     
    if __name__ == '__main__':
        sqlfnm = sys.argv[1]
        outfnm = sys.argv[2]
        sql_to_xsl(sqlfnm, outfnm)
        sys.exit(0)
