--
-- PostgreSQL database dump
--

\restrict oJ7Rb79epf918cNTvyigEYvblKCMVIsJTvq1QOo98FWlNGZTKyY6kGdCslHTH01

-- Dumped from database version 16.10 (Ubuntu 16.10-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.10 (Ubuntu 16.10-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: tsm_system_rows; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS tsm_system_rows WITH SCHEMA public;


--
-- Name: EXTENSION tsm_system_rows; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION tsm_system_rows IS 'TABLESAMPLE method which accepts number of rows as a limit';


SET default_tablespace = shopifystores_space;

SET default_table_access_method = heap;

--
-- Name: builtwith2020; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.builtwith2020 (
    domain character varying,
    "Location on Site" character varying,
    "Tech Spend USD" character varying,
    company character varying,
    vertical character varying,
    quantcast character varying,
    alexa character varying,
    majestic character varying,
    umbrella character varying,
    telephones character varying,
    emails character varying,
    twitter character varying,
    facebook character varying,
    linkedin character varying,
    google character varying,
    pinterest character varying,
    github character varying,
    instagram character varying,
    vk character varying,
    vimeo character varying,
    youtube character varying,
    people character varying,
    city character varying,
    state character varying,
    zip character varying,
    country character varying,
    "First Detected" character varying,
    "Last Found" character varying,
    "First Indexed" character varying,
    "Last Indexed" character varying,
    exclusion character varying,
    gdpr character varying
);


ALTER TABLE public.builtwith2020 OWNER TO postgres;

--
-- Name: builtwith2020_pandas; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.builtwith2020_pandas (
    index bigint,
    domain_name text,
    "Location on Site" text,
    "Tech Spend USD" text,
    "Company" text,
    "Vertical" text,
    "Quantcast" text,
    "Alexa" text,
    "Majestic" text,
    "Umbrella" text,
    "Telephones" text,
    "Emails" text,
    "Twitter" text,
    "Facebook" text,
    "LinkedIn" text,
    "Google" text,
    "Pinterest" text,
    "GitHub" text,
    "Instagram" text,
    "Vk" text,
    "Vimeo" text,
    "Youtube" text,
    "People" text,
    "City" text,
    "State" text,
    "Zip" text,
    "Country" text,
    "First Detected" text,
    "Last Found" text,
    "First Indexed" text,
    "Last Indexed" text,
    "Exclusion" text,
    "GDPR" text
);


ALTER TABLE public.builtwith2020_pandas OWNER TO postgres;

--
-- Name: builtwith_import_failures; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.builtwith_import_failures (
    hostname character varying NOT NULL,
    run_name character varying NOT NULL,
    has_dns_error boolean DEFAULT false NOT NULL,
    dns_error_message character varying,
    ip_address character varying,
    reverse_name_resolution character varying,
    has_shopify_domain boolean DEFAULT false NOT NULL,
    has_metadata_fetch_error boolean DEFAULT false NOT NULL,
    metadata_fetch_message character varying,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.builtwith_import_failures OWNER TO postgres;

--
-- Name: exchange_rates; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.exchange_rates (
    currency_code character varying NOT NULL,
    worth_this_many_usd double precision NOT NULL
);


ALTER TABLE public.exchange_rates OWNER TO postgres;

--
-- Name: metadata; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.metadata (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    metadata jsonb NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.metadata OWNER TO postgres;

--
-- Name: metadata_fetch_failures; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.metadata_fetch_failures (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    problem character varying NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.metadata_fetch_failures OWNER TO postgres;

--
-- Name: multiply_defined_stores; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.multiply_defined_stores (
    hostname character varying NOT NULL,
    myshopify_domain character varying,
    run_name character varying,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.multiply_defined_stores OWNER TO postgres;

SET default_tablespace = shopify_ssd;

--
-- Name: product_details; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopify_ssd
--

CREATE TABLE public.product_details (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    product_handle character varying NOT NULL,
    product_detail jsonb NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    image_metadata_processed boolean DEFAULT false
);


ALTER TABLE public.product_details OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: product_details_fetch_failures; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.product_details_fetch_failures (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    product_handle character varying NOT NULL,
    problem character varying NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.product_details_fetch_failures OWNER TO postgres;

--
-- Name: product_prices; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.product_prices AS
 SELECT metadata.myshopify_domain,
    metadata.run_name,
    (metadata.metadata ->> 'currency'::text) AS currency,
    product_details.product_handle,
    (((((product_details.product_detail -> 'product'::text) -> 'variants'::text) -> 0) ->> 'price'::text))::double precision AS price
   FROM (public.metadata
     JOIN public.product_details USING (myshopify_domain, run_name));


ALTER VIEW public.product_prices OWNER TO postgres;

SET default_tablespace = shopifystores_space;

--
-- Name: product_scouting_fetch_failures; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.product_scouting_fetch_failures (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    problem character varying NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.product_scouting_fetch_failures OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: product_scoutings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.product_scoutings (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    products jsonb NOT NULL,
    when_fetched timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.product_scoutings OWNER TO postgres;

SET default_tablespace = shopifystores_space;

--
-- Name: scouted_products; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE MATERIALIZED VIEW public.scouted_products AS
 SELECT myshopify_domain,
    run_name,
    (jsonb_array_elements((products -> 'products'::text)) ->> 'handle'::text) AS product_handle
   FROM public.product_scoutings
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.scouted_products OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: scrape_runs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scrape_runs (
    run_name character varying NOT NULL,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.scrape_runs OWNER TO postgres;

SET default_tablespace = shopifystores_space;

--
-- Name: shopify_shop_ids; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.shopify_shop_ids (
    shop_id character varying NOT NULL,
    myshopify_domain character varying NOT NULL,
    CONSTRAINT shop_id_domain_name_myshopify_domain_check CHECK (((myshopify_domain)::text ~~ '%.myshopify.com'::text))
);


ALTER TABLE public.shopify_shop_ids OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: sites; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sites (
    myshopify_domain character varying NOT NULL,
    bootstrap_url character varying,
    creation_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT sites_myshopify_domain_check CHECK (((myshopify_domain)::text ~~ '%.myshopify.com'::text))
);


ALTER TABLE public.sites OWNER TO postgres;

SET default_tablespace = shopifystores_space;

--
-- Name: site_recent_history; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE MATERIALIZED VIEW public.site_recent_history AS
 SELECT sites.myshopify_domain,
    max(metadata.when_fetched) AS most_recent_metadata_fetch,
    max(metadata_fetch_failures.when_fetched) AS most_recent_metadata_fetch_failure,
    ((max(metadata.when_fetched) > max(metadata_fetch_failures.when_fetched)) OR ((max(metadata.when_fetched) IS NOT NULL) AND (max(metadata_fetch_failures.when_fetched) IS NULL))) AS currently_alive
   FROM ((public.sites
     LEFT JOIN public.metadata USING (myshopify_domain))
     LEFT JOIN public.metadata_fetch_failures USING (myshopify_domain))
  GROUP BY sites.myshopify_domain
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.site_recent_history OWNER TO postgres;

--
-- Name: store_images; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.store_images (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    product_handle character varying NOT NULL,
    image_id character varying NOT NULL,
    width integer NOT NULL,
    height integer NOT NULL,
    has_alt_text boolean NOT NULL,
    is_hero_image boolean DEFAULT false NOT NULL,
    when_calculated text DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.store_images OWNER TO postgres;

--
-- Name: store_hero_image_summary; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE MATERIALIZED VIEW public.store_hero_image_summary AS
 SELECT myshopify_domain,
    run_name,
    avg((has_alt_text)::integer) AS proportion_with_alt_text,
    count(DISTINCT width) AS number_of_distinct_widths,
    count(DISTINCT height) AS number_of_distinct_heights,
    ((1.0 * (count(DISTINCT width))::numeric) / (count(*))::numeric) AS needs_resize_score,
    count(*) AS number_of_images_in_sample
   FROM public.store_images
  WHERE is_hero_image
  GROUP BY myshopify_domain, run_name
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.store_hero_image_summary OWNER TO postgres;

--
-- Name: store_image_summary; Type: MATERIALIZED VIEW; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE MATERIALIZED VIEW public.store_image_summary AS
 SELECT myshopify_domain,
    run_name,
    avg((has_alt_text)::integer) AS proportion_with_alt_text,
    count(DISTINCT width) AS number_of_distinct_widths,
    count(DISTINCT height) AS number_of_distinct_heights,
    ((1.0 * (count(DISTINCT width))::numeric) / (count(*))::numeric) AS needs_resize_score,
    count(*) AS number_of_images_in_sample
   FROM public.store_images
  GROUP BY myshopify_domain, run_name
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.store_image_summary OWNER TO postgres;

--
-- Name: store_price_summary; Type: MATERIALIZED VIEW; Schema: public; Owner: shopifyscraper; Tablespace: shopifystores_space
--

CREATE MATERIALIZED VIEW public.store_price_summary AS
 SELECT myshopify_domain,
    run_name,
    currency,
    min(price) AS min_price,
    max(price) AS max_price,
    avg(price) AS avg_price,
    stddev(price) AS std_price,
    percentile_cont((0.25)::double precision) WITHIN GROUP (ORDER BY price) AS percentile_25,
    percentile_cont((0.5)::double precision) WITHIN GROUP (ORDER BY price) AS median,
    percentile_cont((0.75)::double precision) WITHIN GROUP (ORDER BY price) AS percentile_75
   FROM public.product_prices
  GROUP BY myshopify_domain, run_name, currency
  WITH NO DATA;


ALTER MATERIALIZED VIEW public.store_price_summary OWNER TO shopifyscraper;

--
-- Name: store_theme_fetch_failures; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.store_theme_fetch_failures (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    failure_message character varying
);


ALTER TABLE public.store_theme_fetch_failures OWNER TO postgres;

--
-- Name: store_themes; Type: TABLE; Schema: public; Owner: postgres; Tablespace: shopifystores_space
--

CREATE TABLE public.store_themes (
    myshopify_domain character varying NOT NULL,
    run_name character varying NOT NULL,
    found_theme_lines boolean,
    theme_name character varying,
    theme_id bigint,
    theme_store_id character varying,
    theme_handle character varying,
    theme_style_id character varying,
    theme_style_handle character varying
);


ALTER TABLE public.store_themes OWNER TO postgres;

--
-- Name: unscraped_products; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.unscraped_products AS
 SELECT scouted_products.myshopify_domain,
    scouted_products.run_name,
    scouted_products.product_handle
   FROM ((public.scouted_products
     LEFT JOIN public.product_details USING (myshopify_domain, run_name, product_handle))
     LEFT JOIN public.product_details_fetch_failures USING (myshopify_domain, run_name, product_handle))
  WHERE ((product_details.myshopify_domain IS NULL) AND (product_details_fetch_failures.myshopify_domain IS NULL));


ALTER VIEW public.unscraped_products OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: builtwith_import_failures builtwith_import_failures_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.builtwith_import_failures
    ADD CONSTRAINT builtwith_import_failures_pkey PRIMARY KEY (hostname);


--
-- Name: exchange_rates exchange_rates_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.exchange_rates
    ADD CONSTRAINT exchange_rates_pkey PRIMARY KEY (currency_code);


--
-- Name: metadata_fetch_failures metadata_fetch_failures_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata_fetch_failures
    ADD CONSTRAINT metadata_fetch_failures_pkey PRIMARY KEY (myshopify_domain, run_name);


--
-- Name: metadata metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata
    ADD CONSTRAINT metadata_pkey PRIMARY KEY (myshopify_domain, run_name);


--
-- Name: multiply_defined_stores multiply_defined_stores_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.multiply_defined_stores
    ADD CONSTRAINT multiply_defined_stores_pkey PRIMARY KEY (hostname);


--
-- Name: product_details_fetch_failures product_details_fetch_failures_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details_fetch_failures
    ADD CONSTRAINT product_details_fetch_failures_pkey PRIMARY KEY (myshopify_domain, run_name, product_handle);


--
-- Name: product_details product_details_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details
    ADD CONSTRAINT product_details_pkey PRIMARY KEY (myshopify_domain, run_name, product_handle);


--
-- Name: product_scouting_fetch_failures product_scouting_fetch_failures_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scouting_fetch_failures
    ADD CONSTRAINT product_scouting_fetch_failures_pkey PRIMARY KEY (myshopify_domain, run_name);


--
-- Name: product_scoutings product_scoutings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scoutings
    ADD CONSTRAINT product_scoutings_pkey PRIMARY KEY (myshopify_domain, run_name);


--
-- Name: scrape_runs scrape_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scrape_runs
    ADD CONSTRAINT scrape_runs_pkey PRIMARY KEY (run_name);


--
-- Name: shopify_shop_ids shop_id_domain_name_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.shopify_shop_ids
    ADD CONSTRAINT shop_id_domain_name_pkey PRIMARY KEY (shop_id);


--
-- Name: sites sites_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sites
    ADD CONSTRAINT sites_pkey PRIMARY KEY (myshopify_domain);


--
-- Name: store_images store_images_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_images
    ADD CONSTRAINT store_images_pkey PRIMARY KEY (myshopify_domain, run_name, product_handle, image_id);


--
-- Name: builtwith2020_pandas_domain_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX builtwith2020_pandas_domain_name_idx ON public.builtwith2020_pandas USING btree (domain_name);


--
-- Name: ix_builtwith2020_pandas_index; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_builtwith2020_pandas_index ON public.builtwith2020_pandas USING btree (index);


--
-- Name: metadata_fetch_failures_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX metadata_fetch_failures_myshopify_domain_idx ON public.metadata_fetch_failures USING btree (myshopify_domain);


--
-- Name: metadata_fetch_failures_myshopify_domain_when_fetched_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX metadata_fetch_failures_myshopify_domain_when_fetched_idx ON public.metadata_fetch_failures USING btree (myshopify_domain, when_fetched);


--
-- Name: metadata_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX metadata_myshopify_domain_idx ON public.metadata USING btree (myshopify_domain);


--
-- Name: metadata_myshopify_domain_when_fetched_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX metadata_myshopify_domain_when_fetched_idx ON public.metadata USING btree (myshopify_domain, when_fetched);


--
-- Name: multiply_defined_stores_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX multiply_defined_stores_myshopify_domain_idx ON public.multiply_defined_stores USING btree (myshopify_domain);


--
-- Name: multiply_defined_stores_myshopify_domain_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX multiply_defined_stores_myshopify_domain_run_name_idx ON public.multiply_defined_stores USING btree (myshopify_domain, run_name);


--
-- Name: product_details_fetch_failure_myshopify_domain_run_name_pro_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX product_details_fetch_failure_myshopify_domain_run_name_pro_idx ON public.product_details_fetch_failures USING btree (myshopify_domain, run_name, product_handle);


--
-- Name: product_details_myshopify_domain_run_name_product_handle_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX product_details_myshopify_domain_run_name_product_handle_idx ON public.product_details USING btree (myshopify_domain, run_name, product_handle) WHERE (image_metadata_processed = false);


--
-- Name: product_details_myshopify_domain_run_name_product_handle_idx1; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX product_details_myshopify_domain_run_name_product_handle_idx1 ON public.product_details USING btree (myshopify_domain, run_name, product_handle);


--
-- Name: product_details_myshopify_domain_run_name_product_handle_im_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX product_details_myshopify_domain_run_name_product_handle_im_idx ON public.product_details USING btree (myshopify_domain, run_name, product_handle, image_metadata_processed);


--
-- Name: raw_builtwith_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX raw_builtwith_domain_idx ON public.builtwith2020 USING btree (domain);


--
-- Name: scouted_products_myshopify_domain_run_name_product_handle_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX scouted_products_myshopify_domain_run_name_product_handle_idx ON public.scouted_products USING btree (myshopify_domain, run_name, product_handle);


--
-- Name: shopify_shop_ids_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX shopify_shop_ids_myshopify_domain_idx ON public.shopify_shop_ids USING btree (myshopify_domain);


--
-- Name: shopify_shop_ids_shop_id_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX shopify_shop_ids_shop_id_idx ON public.shopify_shop_ids USING btree (shop_id);


--
-- Name: shopify_shop_ids_shop_id_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX shopify_shop_ids_shop_id_myshopify_domain_idx ON public.shopify_shop_ids USING btree (shop_id, myshopify_domain);


--
-- Name: site_recent_history_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX site_recent_history_myshopify_domain_idx ON public.site_recent_history USING btree (myshopify_domain) WHERE currently_alive;


--
-- Name: store_hero_image_summary_myshopify_domain_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX store_hero_image_summary_myshopify_domain_run_name_idx ON public.store_hero_image_summary USING btree (myshopify_domain, run_name);


--
-- Name: store_image_summary_myshopify_domain_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX store_image_summary_myshopify_domain_run_name_idx ON public.store_image_summary USING btree (myshopify_domain, run_name);


--
-- Name: store_price_summary_myshopify_domain_run_name_currency_idx; Type: INDEX; Schema: public; Owner: shopifyscraper
--

CREATE UNIQUE INDEX store_price_summary_myshopify_domain_run_name_currency_idx ON public.store_price_summary USING btree (myshopify_domain, run_name, currency);


--
-- Name: store_theme_fetch_failures_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_theme_fetch_failures_myshopify_domain_idx ON public.store_theme_fetch_failures USING btree (myshopify_domain);


--
-- Name: store_theme_fetch_failures_myshopify_domain_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_theme_fetch_failures_myshopify_domain_run_name_idx ON public.store_theme_fetch_failures USING btree (myshopify_domain, run_name);


--
-- Name: store_theme_fetch_failures_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_theme_fetch_failures_run_name_idx ON public.store_theme_fetch_failures USING btree (run_name);


--
-- Name: store_themes_myshopify_domain_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_themes_myshopify_domain_idx ON public.store_themes USING btree (myshopify_domain);


--
-- Name: store_themes_myshopify_domain_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_themes_myshopify_domain_run_name_idx ON public.store_themes USING btree (myshopify_domain, run_name);


--
-- Name: store_themes_theme_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_themes_theme_name_idx ON public.store_themes USING btree (theme_name);


--
-- Name: store_themes_theme_name_run_name_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX store_themes_theme_name_run_name_idx ON public.store_themes USING btree (theme_name, run_name);


--
-- Name: builtwith_import_failures builtwith_import_failures_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.builtwith_import_failures
    ADD CONSTRAINT builtwith_import_failures_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: metadata_fetch_failures metadata_fetch_failures_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata_fetch_failures
    ADD CONSTRAINT metadata_fetch_failures_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: metadata_fetch_failures metadata_fetch_failures_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata_fetch_failures
    ADD CONSTRAINT metadata_fetch_failures_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: metadata metadata_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata
    ADD CONSTRAINT metadata_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: metadata metadata_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metadata
    ADD CONSTRAINT metadata_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: multiply_defined_stores multiply_defined_stores_myshopify; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.multiply_defined_stores
    ADD CONSTRAINT multiply_defined_stores_myshopify FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: product_details_fetch_failures product_details_fetch_failures_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details_fetch_failures
    ADD CONSTRAINT product_details_fetch_failures_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: product_details_fetch_failures product_details_fetch_failures_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details_fetch_failures
    ADD CONSTRAINT product_details_fetch_failures_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: product_details product_details_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details
    ADD CONSTRAINT product_details_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: product_details product_details_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_details
    ADD CONSTRAINT product_details_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: product_scouting_fetch_failures product_scouting_fetch_failures_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scouting_fetch_failures
    ADD CONSTRAINT product_scouting_fetch_failures_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: product_scouting_fetch_failures product_scouting_fetch_failures_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scouting_fetch_failures
    ADD CONSTRAINT product_scouting_fetch_failures_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: product_scoutings product_scoutings_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scoutings
    ADD CONSTRAINT product_scoutings_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: product_scoutings product_scoutings_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.product_scoutings
    ADD CONSTRAINT product_scoutings_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: store_images store_images_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_images
    ADD CONSTRAINT store_images_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: store_images store_images_myshopify_domain_run_name_product_handle_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_images
    ADD CONSTRAINT store_images_myshopify_domain_run_name_product_handle_fkey FOREIGN KEY (myshopify_domain, run_name, product_handle) REFERENCES public.product_details(myshopify_domain, run_name, product_handle);


--
-- Name: store_images store_images_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_images
    ADD CONSTRAINT store_images_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: store_theme_fetch_failures store_theme_fetch_failures_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_theme_fetch_failures
    ADD CONSTRAINT store_theme_fetch_failures_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: store_theme_fetch_failures store_theme_fetch_failures_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_theme_fetch_failures
    ADD CONSTRAINT store_theme_fetch_failures_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: store_themes store_themes_myshopify_domain_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_themes
    ADD CONSTRAINT store_themes_myshopify_domain_fkey FOREIGN KEY (myshopify_domain) REFERENCES public.sites(myshopify_domain);


--
-- Name: store_themes store_themes_run_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.store_themes
    ADD CONSTRAINT store_themes_run_name_fkey FOREIGN KEY (run_name) REFERENCES public.scrape_runs(run_name);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO PUBLIC;
GRANT ALL ON SCHEMA public TO shopifyscraper;


--
-- Name: TABLE builtwith2020; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.builtwith2020 TO chandan;
GRANT SELECT ON TABLE public.builtwith2020 TO rakesh;
GRANT ALL ON TABLE public.builtwith2020 TO shopifyscraper;


--
-- Name: TABLE builtwith2020_pandas; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.builtwith2020_pandas TO chandan;
GRANT SELECT ON TABLE public.builtwith2020_pandas TO rakesh;
GRANT ALL ON TABLE public.builtwith2020_pandas TO shopifyscraper;


--
-- Name: TABLE builtwith_import_failures; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.builtwith_import_failures TO chandan;
GRANT SELECT ON TABLE public.builtwith_import_failures TO rakesh;
GRANT ALL ON TABLE public.builtwith_import_failures TO shopifyscraper;


--
-- Name: TABLE exchange_rates; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.exchange_rates TO chandan;
GRANT SELECT ON TABLE public.exchange_rates TO rakesh;
GRANT ALL ON TABLE public.exchange_rates TO shopifyscraper;


--
-- Name: TABLE metadata; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.metadata TO chandan;
GRANT SELECT ON TABLE public.metadata TO rakesh;
GRANT ALL ON TABLE public.metadata TO shopifyscraper;


--
-- Name: TABLE metadata_fetch_failures; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.metadata_fetch_failures TO chandan;
GRANT SELECT ON TABLE public.metadata_fetch_failures TO rakesh;
GRANT ALL ON TABLE public.metadata_fetch_failures TO shopifyscraper;


--
-- Name: TABLE multiply_defined_stores; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.multiply_defined_stores TO chandan;
GRANT SELECT ON TABLE public.multiply_defined_stores TO rakesh;
GRANT ALL ON TABLE public.multiply_defined_stores TO shopifyscraper;


--
-- Name: TABLE product_details; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.product_details TO chandan;
GRANT SELECT ON TABLE public.product_details TO rakesh;
GRANT ALL ON TABLE public.product_details TO shopifyscraper;


--
-- Name: TABLE product_details_fetch_failures; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.product_details_fetch_failures TO chandan;
GRANT SELECT ON TABLE public.product_details_fetch_failures TO rakesh;
GRANT ALL ON TABLE public.product_details_fetch_failures TO shopifyscraper;


--
-- Name: TABLE product_prices; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.product_prices TO chandan;
GRANT SELECT ON TABLE public.product_prices TO rakesh;
GRANT ALL ON TABLE public.product_prices TO shopifyscraper;


--
-- Name: TABLE product_scouting_fetch_failures; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.product_scouting_fetch_failures TO chandan;
GRANT SELECT ON TABLE public.product_scouting_fetch_failures TO rakesh;
GRANT ALL ON TABLE public.product_scouting_fetch_failures TO shopifyscraper;


--
-- Name: TABLE product_scoutings; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.product_scoutings TO chandan;
GRANT SELECT ON TABLE public.product_scoutings TO rakesh;
GRANT ALL ON TABLE public.product_scoutings TO shopifyscraper;


--
-- Name: TABLE scouted_products; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.scouted_products TO chandan;
GRANT SELECT ON TABLE public.scouted_products TO rakesh;
GRANT ALL ON TABLE public.scouted_products TO shopifyscraper;


--
-- Name: TABLE scrape_runs; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.scrape_runs TO chandan;
GRANT SELECT ON TABLE public.scrape_runs TO rakesh;
GRANT ALL ON TABLE public.scrape_runs TO shopifyscraper;


--
-- Name: TABLE shopify_shop_ids; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.shopify_shop_ids TO chandan;
GRANT SELECT ON TABLE public.shopify_shop_ids TO rakesh;
GRANT ALL ON TABLE public.shopify_shop_ids TO shopifyscraper;


--
-- Name: TABLE sites; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.sites TO chandan;
GRANT SELECT ON TABLE public.sites TO rakesh;
GRANT ALL ON TABLE public.sites TO shopifyscraper;


--
-- Name: TABLE site_recent_history; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.site_recent_history TO chandan;
GRANT SELECT ON TABLE public.site_recent_history TO rakesh;
GRANT ALL ON TABLE public.site_recent_history TO shopifyscraper;


--
-- Name: TABLE store_images; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.store_images TO chandan;
GRANT SELECT ON TABLE public.store_images TO rakesh;
GRANT ALL ON TABLE public.store_images TO shopifyscraper;


--
-- Name: TABLE store_hero_image_summary; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.store_hero_image_summary TO chandan;
GRANT SELECT ON TABLE public.store_hero_image_summary TO rakesh;
GRANT ALL ON TABLE public.store_hero_image_summary TO shopifyscraper;


--
-- Name: TABLE store_image_summary; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.store_image_summary TO chandan;
GRANT SELECT ON TABLE public.store_image_summary TO rakesh;
GRANT ALL ON TABLE public.store_image_summary TO shopifyscraper;


--
-- Name: TABLE store_price_summary; Type: ACL; Schema: public; Owner: shopifyscraper
--

GRANT SELECT ON TABLE public.store_price_summary TO chandan;
GRANT SELECT ON TABLE public.store_price_summary TO rakesh;


--
-- Name: TABLE store_theme_fetch_failures; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.store_theme_fetch_failures TO chandan;
GRANT SELECT ON TABLE public.store_theme_fetch_failures TO rakesh;
GRANT ALL ON TABLE public.store_theme_fetch_failures TO shopifyscraper;


--
-- Name: TABLE store_themes; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.store_themes TO chandan;
GRANT SELECT ON TABLE public.store_themes TO rakesh;
GRANT ALL ON TABLE public.store_themes TO shopifyscraper;


--
-- Name: TABLE unscraped_products; Type: ACL; Schema: public; Owner: postgres
--

GRANT SELECT ON TABLE public.unscraped_products TO chandan;
GRANT SELECT ON TABLE public.unscraped_products TO rakesh;
GRANT ALL ON TABLE public.unscraped_products TO shopifyscraper;


--
-- PostgreSQL database dump complete
--

\unrestrict oJ7Rb79epf918cNTvyigEYvblKCMVIsJTvq1QOo98FWlNGZTKyY6kGdCslHTH01

