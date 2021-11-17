$(document).ready(function() {

    $('form').on('submit', function(event) {

        $.ajax({
            data: {
                weight : $('#weight').val(),
                vis : $('#vis').val(),
                mrp : $('#mrp').val(),
                years : $('#years').val(),
                fat : $('#fat').val(),
                item_type : $('#item_type').val(),
                outlet : $('#outlet').val(),
                size : $('#size').val(),
                tier : $('#tier').val(),
                outlet_type : $('#outlet_type').val(),
                ml : $('#ml').val()
            },
            type: 'POST',
            url: '/predict'
        })
        .done(function(data) {
            
            if (data.prediction) {
                $('#hide').show();
                $('#predict').text("$ "+data.prediction).show();
            }
            else {
                $('#hide').hide();
            }

        });

        event.preventDefault();
    });
});