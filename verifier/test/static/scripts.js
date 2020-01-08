// text to permission
const t2p_module = (function () {

    const PERMISSIONS = ["calendar", "call_log", "camera", "contacts", "location",
        "microphone", "phone", "sms", "ext_storage"];
    const COLOR = {
        green: '144, 201, 120',
        red: '255, 105, 97',
        grey: '212, 212, 200',
        blue: '0, 123, 255'
    };
    const PERMISSION_ICONS = {
        "calendar": "&#xe916;",
        "phone": "&#xe0b0;",
        "ext_storage": "&#xe161;",
        "call_log": "&#xe040;",
        "camera": "&#xe412;",
        "sms": "&#xe0c9;",
        "microphone": "&#xe31d;",
        "location": "&#xe0c8;",
        "contacts": "&#xe0cf;"
    };
    const THRESHOLD_TRUE = 0.75;
    const THRESHOLD_FALSE = 0.25;

    let loaded = {};
    let permissions_actual = {};
    let currentFile = null;

    // ----------------------------------

    function show() {
        $('#panel_t2p').show();
    }

    function hide() {
        $('#panel_t2p').hide();
    }


    function load(file, callback) {
        if (currentFile === file) {
            return callback(true);
        }
        currentFile = file;

        $.getJSON(file).done(loaded_data => {
            callback(true);
            initHeader(loaded_data);

            loaded = loaded_data['t2p'] || {};
            permissions_actual = loaded_data['permissions_actual'];

            highlightText(null);
            initButtons();
        }).fail(() => callback(false));
    }

    function initHeader(header_data) {
        if (header_data['app_icon'] && header_data['app_icon'].length > 0) {
            $('#app_icon').attr('src', header_data['app_icon']);
            $('#app_icon_col').show();
        } else {
            $('#app_icon_col').hide();
        }

        $('#app_name').text(header_data['app_name'] || "Unknown App");

        //$('#dangerous_permissions').text(loaded['dangerous_permissions'] || "?");
        $('#package').text(header_data['package'] || "- missing - ");
        $('#version').text(header_data['version'] || "- missing - ");
    }


    function initButtons() {
        $('#permissions').html("");

        $('#permissions').append($('<h3>Requested Permissions</h3>'));

        let in_category = false;

        for (let i = 0; i < PERMISSIONS.length; i++) {
            if (permissions_actual[PERMISSIONS[i]] === false) continue;
            const pct = loaded.permissions_pred[PERMISSIONS[i]];
            const permission_name = PERMISSIONS[i];
            $('#permissions').append(createPermissionListItem(permission_name, pct));
            in_category = true;
        }

        if (!in_category) {
            $('#permissions').append($('<li class="list-group-item text-small">None</li>'));
        }

        $('#permissions').append($('<h3 class="mt-1">Not requested</h3>'));

        for (let i = 0; i < PERMISSIONS.length; i++) {
            if (permissions_actual[PERMISSIONS[i]] === true) continue;
            const pct = loaded.permissions_pred[PERMISSIONS[i]];
            const permission_name = PERMISSIONS[i];
            $('#permissions').append(createPermissionListItem(permission_name, pct));
        }
    }

    function createPermissionListItem(permission_name, pct) {
        const pct_predicted_bool = (pct > THRESHOLD_TRUE) ? true : (pct < THRESHOLD_FALSE) ? false : null;

        let badge_class = 'badge-secondary';
        if (pct_predicted_bool === permissions_actual[permission_name]) {
            badge_class = 'badge-success';
        } else if (pct_predicted_bool === null) {
            badge_class = 'badge-warning';
        } else {
            badge_class = 'badge-danger';
        }
        pct = Math.min(0.9, pct);
        pct = "" + Math.round(pct * 100) + "%";

        const li = $('.template_permission_list').clone();
        li.css({display: 'block'});
        if (pct) {
            li.find('.badge').addClass(badge_class);
            li.find('.badge').text(pct);
        } else {
            li.find('.badge').hide();
        }
        li.find('i').html(PERMISSION_ICONS[permission_name]);
        li.removeClass('template_permission_list');
        li.find('h6 a').text(permission_name);
        li.click(e => {
            $('#permissions li').removeClass("active");
            $(e.target).parents('li').addClass("active");
            highlightText(permission_name);
        });

        return li;
    }

    function highlightText(permission) {
        let tokens_and_importance = loaded.tokens.map(t => [t, 0.]);

        let pairs = [];
        if (permission == null) {
            const tokens_ids = {};
            for (let i = 0; i < PERMISSIONS.length; i++) {
                const p_pairs = loaded.tokens_heat[PERMISSIONS[i]] || [];
                for (let j = 0; j < p_pairs.length; j++) {
                    const idx = p_pairs[j][0];
                    const heat = p_pairs[j][1];
                    tokens_ids[idx] = Math.max(heat, tokens_ids[idx] || 0);
                }
            }
            pairs = Object.keys(tokens_ids).map(idx => [idx, tokens_ids[idx]]);
        } else {
            pairs = loaded.tokens_heat[permission] || [];
        }

        console.log("pairs", pairs);

        for (let i = 0; i < pairs.length; i++) {
            const idx = parseInt(pairs[i][0]);
            tokens_and_importance[idx][1] = pairs[i][1];
        }

        tokens_and_importance = tokens_and_importance.map(pair => {
           if(pair[1] === 0.) {
               const first_token_occurrence = tokens_and_importance.filter(t => t[0] === pair[0]);
               if(first_token_occurrence.length > 0) {
                   pair[1] = first_token_occurrence[0][1];
               }
           }
           return pair;
        });

        console.log("tokens_and_importance", tokens_and_importance);

        const bg = permission ? COLOR.blue : COLOR.grey;

        const text_container = $('#text_container');
        text_container.html("");
        for (let i = 0; i < tokens_and_importance.length; i++) {
            const span = $('<span>');
            span.css({backgroundColor: 'rgba(' + bg + ', ' + tokens_and_importance[i][1]/100 + ')'});
            let token = tokens_and_importance[i][0];
            token = token === ' ' ? '&nbsp;' : token;
            token = token === '\n' ? '<br />' : token;
            span.html(tokens_and_importance[i][0]);
            text_container.append(span);
        }
    }

    return {
        load: load,
        show: show,
        hide: hide
    }

});

// main module
const main = (function () {
    let _list = [];

    const t2p = t2p_module();

    let active_file = null;
    let _toast_timeout = null;

    function onDocumentLoad() {
        loadFilesList();
        $("#menu-toggle").click((e) => {
            e.preventDefault();
            $("#wrapper").toggleClass("toggled");
        });
        toast('Loaded random app. Use the menu to see other results.');
    }

    function loadFilesList(callback) {
        $.getJSON('list.json', function (obj) {
            _list = obj || [];

            if (!_list || !_list.length) {
                error();
                return
            }

            _list.sort((a, b) => {
                a = a.title;
                b = b.title;
                if (a.toLowerCase() < b.toLowerCase()) return -1;
                if (a.toLowerCase() > b.toLowerCase()) return 1;
                return 0;
            });

            const random_app = _list[parseInt(Math.random() * _list.length)];
            activateAndLoad(random_app.file, null, callback);

            $('#apps_list').html("");
            _list.forEach(element => {
                const el = $('#template_app_list').clone();
                el.find(".title").text(element.title);
                if (element.t2p) {
                    el.find(".btn_permissions").click(() => {
                        activateAndLoad(element.file, null);
                    });
                }
                el.css({display: 'block'});
                $('#apps_list').append(el);
            })
        });
    }

    function error() {
        $('#main_content_container').hide();
        $('#main_loading').html('<div class="alert alert-danger" role="alert">\n' +
            '  <h4 class="alert-heading">Error occured</h4>\n' +
            '  <p>The given app result files seem to be invalid.</p>\n' +
            '</div>');
    }

    function showMain() {
        $('#main_content_container').show();
        $('#main_loading').hide();
    }

    function activateAndLoad(file, mode, callback) {
        $('#main_loading').show();
        $('#main_content_container').hide();
        active_file = file;

        const callbackHandler = success => {
            handleFileLoadResult(success);
            if(callback) callback(success);
        };

        t2p.show();
        t2p.load('apps/' + file, callbackHandler);

        if ($('#apps_list').is(':visible')) {
            $('button.navbar-toggler').click();
        }

    }

    function handleFileLoadResult(success) {
        if (!success) {
            error();
        } else {
            showMain();
        }
    }

    $(document).ready(() => {
        window.setTimeout(onDocumentLoad, 800);

        $(document).keyup(function (e) {
            if (e.which !== 37 && e.which !== 39) return;

            const direction = (e.which === 37) ? -1 : 1;
            const idx_current = _list.map(f => f.file).indexOf(active_file);
            const idx_new = idx_current + direction;

            if (idx_new >= 0 && idx_new < _list.length) {
                activateAndLoad(_list[idx_new].file, null);
                toast('App ' + (idx_new + 1) + ' of ' + _list.length);
            } else {
                toast('No more apps');
            }
        });
    });

    function toast(text) {
        const minitoast = $('#minitoast');
        minitoast.find('#minitoast_text').text(text);
        minitoast.fadeIn();
        clearTimeout(_toast_timeout);
        _toast_timeout = setTimeout(() => minitoast.fadeOut(), 3500);
    }

    return {
        toast: toast
    }

})();
